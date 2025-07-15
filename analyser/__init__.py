import logging
import os
import re
import requests
from datetime import datetime
from pymongo import MongoClient
from openai import AzureOpenAI, OpenAIError
import azure.functions as func

# Load environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")
COSMOS_CONN_STRING = os.getenv("COSMOS_CONN_STRING")

# GitHub headers
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def strip_markdown(text):
    """Remove markdown to simplify parsing"""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"#+ ", "", text)
    return text.lower()

def validate_environment():
    required_vars = {
        "GITHUB_TOKEN": GITHUB_TOKEN,
        "GITHUB_REPOSITORY": GITHUB_REPO,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
        "COSMOS_CONN_STRING": COSMOS_CONN_STRING
    }
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    logging.info("Environment validated successfully.")

def get_open_prs():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls?state=open"
    logging.info(f"Fetching open PRs from {GITHUB_REPO}")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    prs = resp.json()
    logging.info(f"Retrieved {len(prs)} open PR(s)")
    return prs

def get_pr_file_diffs(pr_number):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls/{pr_number}/files"
    logging.info(f"Fetching file diffs for PR #{pr_number}")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return [(f["filename"], f.get("patch", "")) for f in resp.json()]

def generate_gpt_analysis(target_pr, other_prs):
    logging.info(f"Generating GPT analysis for PR #{target_pr['number']}")

    prompt = f"""You are a code reviewer. Compare the following pull requests for potential conflicts, overlaps, or dependencies.

Target PR #{target_pr['number']} - {target_pr['title']}
Compare with:
{chr(10).join([f"- PR #{pr['number']} - {pr['title']}" for pr in other_prs])}

Please identify:
- Conflicts (same function or line modified)
- Dependencies (if one PR relies on another)
- Safe merges (independent changes)

Be concise and clear in your analysis.
"""

    all_prs = [target_pr] + other_prs
    for pr in all_prs:
        prompt += f"\n\nPR #{pr['number']}:\n"
        file_diffs = get_pr_file_diffs(pr['number'])
        for filename, patch in file_diffs:
            truncated_patch = patch[:500] if patch else "(No diff available)"
            prompt += f"\nFile: {filename}\n```\n{truncated_patch}\n```"

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.5
        )
        result = response.choices[0].message.content.strip()
        logging.info("GPT analysis complete.")
        logging.debug("GPT Output:\n" + result)
        return result
    except OpenAIError as e:
        error_message = f"Azure OpenAI API Error: {str(e)}"
        logging.error(error_message)
        return error_message

def post_comment_on_pr(pr_number, comment_body):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{pr_number}/comments"
    logging.info(f"Posting comment to PR #{pr_number}")
    resp = requests.post(url, headers=HEADERS, json={"body": comment_body})
    if resp.status_code == 201:
        logging.info(f"Comment posted to PR #{pr_number}")
    else:
        logging.error(f"Failed to post comment: {resp.text}")

def add_label_to_pr(pr_number, label):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{pr_number}/labels"
    logging.info(f"Adding label '{label}' to PR #{pr_number}")
    resp = requests.post(url, headers=HEADERS, json={"labels": [label]})
    if resp.status_code in (200, 201):
        logging.info(f"Label '{label}' added to PR #{pr_number}")
    else:
        logging.error(f"Failed to add label: {resp.status_code} - {resp.text}")

def post_to_teams(message):
    if not TEAMS_WEBHOOK_URL:
        logging.warning("TEAMS_WEBHOOK_URL not set. Skipping Teams notification.")
        return
    payload = {
        "title": "Azure OpenAI PR Conflict Analysis",
        "text": message
    }
    try:
        resp = requests.post(TEAMS_WEBHOOK_URL, json=payload)
        if resp.status_code == 200:
            logging.info("Notification sent to Microsoft Teams.")
        else:
            logging.error(f"Failed to send Teams notification: {resp.text}")
    except Exception as e:
        logging.exception("Error sending message to Teams")

def store_pr_to_cosmos(pr_data):
    try:
        client = MongoClient(COSMOS_CONN_STRING)
        db = client["pranalysisdb"]
        collection = db["prs"]
        existing = collection.find_one({"pr_number": pr_data["pr_number"]})
        if existing:
            collection.replace_one({"_id": existing["_id"]}, pr_data)
            logging.info(f"Updated PR #{pr_data['pr_number']}")
        else:
            collection.insert_one(pr_data)
            logging.info(f"Inserted PR #{pr_data['pr_number']} into Cosmos DB")
    except Exception as e:
        logging.exception("Failed to insert PR data into Cosmos DB")

def main_logic():
    validate_environment()

    prs = get_open_prs()
    if len(prs) < 2:
        return "Need at least 2 open PRs to analyze."

    target_pr = prs[0]
    other_prs = prs[1:3]

    gpt_output = generate_gpt_analysis(target_pr, other_prs)

    if not gpt_output.startswith("Azure OpenAI API Error"):
        comment = f"Azure OpenAI Conflict Check\n\n{gpt_output}"
        post_comment_on_pr(target_pr['number'], comment)
        post_to_teams(comment)

    summary = []
    clean_output = strip_markdown(gpt_output)

    has_conflict = "conflicts" in clean_output and "no conflicts" not in clean_output
    has_dependency = "dependencies" in clean_output and "no dependencies" not in clean_output
    has_safe_merge = (
        ("safe merges" in clean_output or "can be merged safely" in clean_output)
        and "no conflicts" in clean_output and "no dependencies" in clean_output
    )

    if has_conflict:
        summary.append("Conflict Detected")
    if has_dependency:
        summary.append("Dependency Found")
    if has_safe_merge and not has_conflict and not has_dependency:
        summary.append("Safe to Merge")
    if not summary:
        summary.append("No Analysis Result")

    flat_summary = " | ".join(summary)
    logging.info(f"Pipeline Summary: {flat_summary}")

    # Apply label based on risk level
    if has_conflict:
        add_label_to_pr(target_pr["number"], "risky")
    elif has_dependency:
        add_label_to_pr(target_pr["number"], "needs-review")
    elif has_safe_merge:
        add_label_to_pr(target_pr["number"], "safe-to-merge")
    else:
        add_label_to_pr(target_pr["number"], "needs-review")

    # Store PR metadata in Cosmos DB
    pr_info = {
        "pr_number": target_pr["number"],
        "title": target_pr["title"],
        "author": target_pr["user"]["login"],
        "files_changed": [f[0] for f in get_pr_file_diffs(target_pr["number"])],
        "gpt_analysis": gpt_output,
        "conflicts_with": [pr["number"] for pr in other_prs if "conflict" in gpt_output.lower()],
        "dependencies": [pr["number"] for pr in other_prs if "dependency" in gpt_output.lower()],
        "safe_to_merge": has_safe_merge,
        "created_at": target_pr.get("created_at", datetime.utcnow().isoformat())
    }

    store_pr_to_cosmos(pr_info)
    return f"AI Review Summary: {flat_summary}"

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Azure Function: PR Analyzer triggered.')
    try:
        result = main_logic()
        return func.HttpResponse(result, status_code=200)
    except Exception as e:
        logging.exception("Azure Function failed")
        return func.HttpResponse(f"Function error: {repr(e)}", status_code=500)
