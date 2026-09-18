# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper script to gather git commits and PRs for changelog updates."""

import argparse
import os
import re
import subprocess
from typing import Any, Dict, List, Optional


REPO_URL = "https://github.com/AI-Hypercomputer/maxtext"

# Categories mapped by conventional commit type or keyword heuristics
CATEGORY_KEYWORDS = {
    "Post-Training": [
        "rl",
        "grpo",
        "dpo",
        "sft",
        "lora",
        "qlora",
        "vllm",
        "post-training",
        "post_training",
        "post_train",
        "tunix",
        "raiden",
        "distill",
        "rollout",
    ],
    "Performance": [
        "fp8",
        "fp4",
        "int8",
        "int4",
        "quant",
        "qwix",
        "aqt",
        "gemm",
        "matmul",
        "throughput",
        "perf",
        "tflops",
        "overlap",
        "collective",
        "double-buffer",
        "memory",
        "muon",
        "sparse_core",
        "sparsecore",
    ],
    "Checkpointing / Goodput": [
        "checkpoint",
        "orbax",
        "goodput",
        "elasticity",
        "restore",
        "slice",
        "multi-tier",
        "resilience",
        "zarr",
        "ocdbt",
    ],
    "Pre-Training": [
        "deepseek",
        "qwen",
        "gemma",
        "llama",
        "mixtral",
        "gpt",
        "moe",
        "diffusion",
        "vit",
        "mamba",
        "decoder",
        "encoder",
        "architecture",
        "model",
        "attention",
        "splash",
        "ring",
        "rope",
        "mrope",
        "mtp",
        "pre_train",
        "pre-train",
        "diloco",
        "nnx",
        "linen",
    ],
    "Usability": [
        "wandb",
        "grain",
        "eval",
        "tutorial",
        "doc",
        "guide",
        "config",
        "pypi",
    ],
}


EXCLUDED_PATHSPECS = [
    ":(exclude)tests/",
    ":(exclude).github/workflows/",
    ":(exclude).github/",
    ":(exclude).gemini/",
    ":(exclude)src/maxtext/training_engine/",
    ":(exclude)**/training_engine/**",
    ":(exclude)src/maxtext/experimental/",
    ":(exclude)**/experimental/**",
    ":(exclude)experimental/",
    ":(exclude)src/dependencies/dockerfiles/",
    ":(exclude)tools/dev/",
    ":(exclude)README.md",
    ":(exclude)docs/release_notes.md",
]

EXCLUDED_PATH_PREFIXES = (
    "tests/",
    ".github/workflows/",
    ".github/",
    ".gemini/",
    "src/maxtext/training_engine/",
    "src/maxtext/experimental/",
    "experimental/",
    "src/dependencies/dockerfiles/",
    "src/dependencies/scripts/",
    "tools/dev/",
)


def is_excluded_path(path: str) -> bool:
  """Returns True if a file path belongs to an excluded directory or file type."""
  p = path.strip().lstrip("./")
  if not p:
    return True
  if p.startswith(EXCLUDED_PATH_PREFIXES):
    return True
  if any(seg in f"/{p}" for seg in ("/training_engine/", "/.github/workflows/", "/experimental/")):
    return True
  if p.endswith(".Dockerfile") or p.endswith("Dockerfile") or p in ("pytest.ini", "README.md", "docs/release_notes.md"):
    return True
  return False


def get_head_commit(repo_dir: str = ".") -> str:
  """Returns the current HEAD commit hash."""
  result = subprocess.run(
      ["git", "-C", repo_dir, "rev-parse", "--short=9", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
  )
  return result.stdout.strip()


def get_last_updated_commit(repo_dir: str = ".") -> str:
  """Reads `**Last Updated**: <commit_hash>` from docs/release_notes.md."""
  path = os.path.join(repo_dir, "docs", "release_notes.md")
  if os.path.exists(path):
    try:
      with open(path, "r", encoding="utf-8") as f:
        content = f.read()
      match = re.search(r"\*{0,2}Last Updated\*{0,2}\s*:\s*\*{0,2}`?([0-9a-f]{7,40})`?", content, re.IGNORECASE)
      if match:
        commit_hash = match.group(1)
        check = subprocess.run(
            ["git", "-C", repo_dir, "cat-file", "-e", f"{commit_hash}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode == 0:
          return commit_hash
    except OSError:
      pass

  return None


def run_git_log(last_commit: str, repo_dir: str = ".") -> str:
  """Runs git log --first-parent for <last_commit>..HEAD with excluded pathspecs."""
  delimiter = "===COMMIT_DELIMITER==="
  field_sep = "===FIELD_SEP==="
  files_sep = "===FILES_SEP==="
  # %H: hash, %P: parent hashes, %an: author, %ad: author date, %s: subject, %b: body
  pretty_format = f"{delimiter}%H{field_sep}%P{field_sep}%an{field_sep}%ad{field_sep}%s{field_sep}%b{files_sep}"
  cmd = [
      "git",
      "-C",
      repo_dir,
      "log",
      "--first-parent",
      "-m",
      "--name-only",
      "--date=short",
      f"--pretty=format:{pretty_format}",
      f"{last_commit}..HEAD",
      "--",
      ".",
      *EXCLUDED_PATHSPECS,
  ]
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)
  return result.stdout


def is_followup_or_cleanup_subject(subject: str) -> bool:
  """Returns True if a commit subject is a PR review follow-up, comment cleanup, or lint fix."""
  s = subject.strip().lower()
  # Strip conventional commit prefix if present
  conv_match = re.match(r"^[a-z]+(?:\([^)]+\))?!?:\s*(.*)", s)
  desc = conv_match.group(1).strip() if conv_match else s

  followup_patterns = [
      r"^address(?:ed|ing)?\s+(?:code\s+)?(?:review|pr)\s+(?:feedback|comments?)",
      r"^review\s+feedback",
      r"^pr\s+feedback",
      r"^address(?:ed|ing)?\s+comments?",
      r"^clean\s*up\s+(?:comments?|descriptions?|docstrings?|code|formatting|imports?)",
      r"^fix(?:ed)?\s+(?:pylint|pyink|lint(?:er|ing)?|format(?:ting)?|typos?|nit(?:s)?|docs?)\b",
      r"^(?:pylint|pyink|lint|format|formatting|reformat|nit|typo)\b",
      r"^update\s+(?:src|tests|docs)/",
      r"^merge\s+(?:branch|remote-tracking\s+branch)\b",
      r"^(?:add|fix|improve)\s+(?:tests?|testing)\b",
      r"^reverts?\s+[0-9a-f]{7,40}",
      r"^no\s+public\s+description\b",
  ]
  return any(re.search(pat, desc) or re.search(pat, s) for pat in followup_patterns)


def normalize_subject(subject: str, body: str = "") -> str:
  """Cleans up commit subjects (extracts Copybara import titles and strips inline bullet lists)."""
  s = subject.strip()
  if s.lower().startswith("copybara import of the project"):
    # Look for the first non-empty line after '<hash> by <author>:'
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and ln.strip() != "--"]
    for i, ln in enumerate(lines):
      if re.match(r"^[0-9a-f]{7,40}\s+by\s+", ln) and i + 1 < len(lines):
        s = lines[i + 1]
        break
  s = re.sub(r"^#+\s*description\s*:?\s*", "", s, flags=re.IGNORECASE).strip()
  # If a single-paragraph commit message includes bullet points (` - `), keep only the headline
  s = re.split(r"\s+-\s+(?=[A-Z`'])", s, maxsplit=1)[0].strip()
  return s


def is_excluded_commit(subject: str, body: str = "", scope: str = "") -> bool:
  """Returns True if a commit should be completely excluded from release notes."""
  s = normalize_subject(subject, body).lower()
  text = f"{s} {body.lower()}"
  sc = scope.strip().lower()
  conv_match = re.match(r"^([a-z]+)(?:\(([^)]+)\))?!?:\s*(.*)", s)
  commit_type = conv_match.group(1).strip().lower() if conv_match else ""
  if conv_match:
    if not sc and conv_match.group(2):
      sc = conv_match.group(2).strip().lower()
    desc = conv_match.group(3).strip()
  else:
    desc = s

  # 1. Exclude follow-up review feedback, comment cleanup, formatting, or minor chore commits
  if is_followup_or_cleanup_subject(s):
    return True
  if (
      commit_type in ("style", "chore", "test", "ci", "build")
      or desc in ("update", "fix linter issues", "lint", "format", "no public description")
      or desc.startswith("reverts ")
      or desc.startswith("suppress new pyrefly")
      or desc.startswith("add auto_gha prefix")
  ) and not any(kw in text for kw in ["support", "upgrade", "enable", "migration", "release", "tutorial"]):
    return True

  # 2. Exclude training_engine, experimental, CI/workflows, test scopes, or docker/container dependency scopes
  if any(
      tok in sc
      for tok in (
          "training_engine",
          "maxtext_engine",
          "experimental",
          "test",
          "tests",
          "workflow",
          "workflows",
          "ci",
          "gha",
      )
  ):
    return True
  if sc in ("dependencies", "docker", "container") and any(
      kw in s for kw in ("image", "container", "docker", "vulnerabilit", "cve", "artifact registry")
  ):
    return True

  # 3. Exclude specific subjects/topics:
  excluded_patterns = [
      r"^no\s+public\s+description\b",
      r"\btraining_engine\b",
      r"\bmaxtexttrainingengine\b",
      r"\bmaxtext_engine\b",
      r"\bengine\s+packing\s+path\b",
      r"\bcontainer\s+image\s+vulnerabilit",
      r"\breduce\s+.*vulnerabilit",
      r"\bmigrate\s+.*docker\s+images?\b",
      r"\bdocker\s+images?\s+to\s+artifact\s+registry\b",
      r"\bdockerfiles?\b",
      r"\bgemini\s+skill\b",
      r"\bupdate[- ]changelog\b",
      r"\bskill\s+to\s+update\s+changelog\b",
      r"\.github/workflows",
      r"\bgithub\s+workflows?\b",
      r"\bpytest[- ]split\b",
      r"\bsplit\s+heavy\s+test\s+jobs\b",
      r"\btest\s+jobs\s+across\s+more\s+workers\b",
      r"\bin\s+latest\s+news\b",
      r"^fix(?:ed|es)?\s+(?:[a-z0-9_.-]+\s+){0,2}tests?\b",
      r"^fix(?:ed|es)?\s+test_[a-z0-9_]+",
  ]
  return any(re.search(pat, s) or re.search(pat, desc) for pat in excluded_patterns)


def resolve_pr_branch_commit(repo_dir: str, parent1: str, parent2: str) -> Optional[Dict[str, str]]:
  """Finds the primary feature/fix commit on a merged PR branch (parent1..parent2), excluding excluded paths."""
  delimiter = "===BRANCH_COMMIT==="
  field_sep = "===BRANCH_FIELD==="
  cmd = [
      "git",
      "-C",
      repo_dir,
      "log",
      f"{parent1}..{parent2}",
      "--no-merges",
      "--reverse",
      f"--pretty=format:{delimiter}%s{field_sep}%b",
      "--",
      ".",
      *EXCLUDED_PATHSPECS,
  ]
  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if result.returncode != 0 or not result.stdout.strip():
    return None

  candidates = []
  for raw in result.stdout.split(delimiter):
    raw = raw.strip()
    if not raw:
      continue
    parts = raw.split(field_sep, 1)
    subj = parts[0].strip()
    body = clean_body_text(parts[1]) if len(parts) > 1 else ""
    if subj:
      candidates.append({"subject": subj, "body": body})

  if not candidates:
    return None

  # Prefer the first commit on the PR branch that is not a review-feedback/lint/cleanup commit
  for cand in candidates:
    if not is_followup_or_cleanup_subject(cand["subject"]):
      return cand

  return candidates[0]


def extract_pr_number(subject: str, body: str) -> Optional[str]:
  """Extracts PR number from commit subject or body."""
  merge_match = re.search(r"Merge pull request #(\d+)", subject)
  if merge_match:
    return merge_match.group(1)
  paren_match = re.search(r"\(#(\d+)\)", subject)
  if paren_match:
    return paren_match.group(1)
  body_match = re.search(r"(?:PR|pull request|#) ?#?(\d{4,5})\b", body, re.IGNORECASE)
  if body_match:
    return body_match.group(1)
  return None


def clean_body_text(body: str) -> str:
  """Removes PiperOrigin-RevId and other metadata noise from commit body."""
  lines = []
  for line in body.splitlines():
    if line.strip().startswith("PiperOrigin-RevId:"):
      continue
    if line.strip().startswith("COPYBARA_INTEGRATE_REVIEW="):
      continue
    lines.append(line)
  return "\n".join(lines).strip()


def is_deprecation_commit(subject: str, body: str = "") -> bool:
  """Returns True if a commit subject deprecates or removes legacy/deprecated flags, APIs, or modules."""
  s = subject.strip().lower()
  deprecation_patterns = [
      r"\bdeprecat(?:e|ed|es|ing|ion)\b",
      r"\[nnx\]\s*delete\s+linen",
      r"\b(?:delete|remove)\s+(?:legacy|deprecated|obsolete)\b",
      r"\bremove\s+the\s+pure_nnx\b",
      r"\bremove\s+linen\b",
      r"\bremove\s+to_huggingface\.py\b",
      r"\bremove\s+`?--[a-z0-9_-]+`?\s+flag\b",
      r"\bdrop\s+support\s+for\b",
  ]
  return any(re.search(pat, s) for pat in deprecation_patterns)


def classify_commit(subject: str, body: str) -> Dict[str, Any]:
  """Classifies a commit into a section (Changes, Bug Fixes, or Deprecations) and category."""
  clean_subject = re.sub(r"^Merge pull request #\d+ from [^\s]+\s*", "", subject).strip()
  if not clean_subject and body:
    clean_subject = body.strip().splitlines()[0]
  clean_subject = normalize_subject(clean_subject, body)

  conv_match = re.match(r"^([a-z]+)(?:\(([^)]+)\))?!?:\s*(.*)", clean_subject, re.IGNORECASE)
  commit_type = conv_match.group(1).lower() if conv_match else ""
  scope = conv_match.group(2).lower() if conv_match and conv_match.group(2) else ""
  description = conv_match.group(3) if conv_match else clean_subject

  is_deprecation = commit_type in ("deprecate", "deprecated") or is_deprecation_commit(clean_subject, body)

  is_bugfix = not is_deprecation and (
      commit_type in ("fix", "bugfix", "hotfix")
      or clean_subject.lower().startswith("fix ")
      or clean_subject.lower().startswith("fixed ")
      or "fix:" in clean_subject.lower()
      or "bug" in scope
  )

  matched_category = "Usability"
  subject_search = f"{scope} {clean_subject}".lower()
  body_search = body.lower()
  matched = False
  for category, keywords in CATEGORY_KEYWORDS.items():
    if any(re.search(rf"\b{re.escape(kw)}\b", subject_search) for kw in keywords):
      matched_category = category
      matched = True
      break
  if not matched:
    for category, keywords in CATEGORY_KEYWORDS.items():
      if any(re.search(rf"\b{re.escape(kw)}\b", body_search) for kw in keywords):
        matched_category = category
        break

  if is_deprecation:
    section = "Deprecations"
  elif is_bugfix:
    section = "Bug Fixes"
  else:
    section = "Changes"

  return {
      "clean_subject": description or clean_subject,
      "commit_type": commit_type or ("deprecate" if is_deprecation else ("fix" if is_bugfix else "feat")),
      "scope": scope,
      "section": section,
      "category": matched_category,
  }


def parse_commits(raw_log: str, repo_dir: str = ".") -> List[Dict[str, Any]]:
  """Parses raw git log --first-parent output and resolves merge commits to their primary PR commit."""
  delimiter = "===COMMIT_DELIMITER==="
  field_sep = "===FIELD_SEP==="
  files_sep = "===FILES_SEP==="

  commits = []
  seen_prs = set()
  seen_hashes = set()

  for entry in raw_log.split(delimiter):
    entry = entry.strip()
    if not entry:
      continue
    parts = entry.split(field_sep)
    if len(parts) < 6:
      continue
    commit_hash = parts[0].strip()
    parents = parts[1].strip().split()
    author = parts[2].strip()
    date = parts[3].strip()
    subject = parts[4].strip()

    body_and_files = parts[5].split(files_sep, 1)
    raw_body = body_and_files[0]
    files_block = body_and_files[1] if len(body_and_files) > 1 else ""
    changed_files = [line.strip() for line in files_block.splitlines() if line.strip()]

    # Exclude if all changed files are in excluded paths (tests/, .github/workflows/, training_engine/, etc.)
    # or if the majority of non-test changed files are in excluded directories
    if changed_files:
      non_test_files = [f for f in changed_files if not f.startswith("tests/")]
      if not non_test_files:
        continue
      excluded_count = sum(1 for f in non_test_files if is_excluded_path(f))
      if excluded_count == len(non_test_files) or excluded_count * 2 >= len(non_test_files):
        continue

    body = clean_body_text(raw_body)

    pr_num = extract_pr_number(subject, raw_body)
    is_merge = subject.startswith("Merge pull request #") or len(parents) > 1

    if is_merge and len(parents) >= 2:
      # If merge commit body has a non-empty title (e.g. GitHub default merge message), use it
      # unless it's empty or a review-feedback/cleanup message, in which case inspect parent1..parent2
      first_body_line = body.splitlines()[0].strip() if body else ""
      if first_body_line and not is_followup_or_cleanup_subject(first_body_line):
        subject = first_body_line
      else:
        resolved = resolve_pr_branch_commit(repo_dir, parents[0], parents[1])
        if resolved:
          subject = resolved["subject"]
          body = resolved["body"] or body
        elif subject.startswith("Merge pull request #"):
          branch_match = re.search(r"from [^:]+:(.+)$", subject)
          if branch_match:
            subject = branch_match.group(1).replace("-", " ").replace("_", " ")

    if commit_hash in seen_hashes:
      continue
    seen_hashes.add(commit_hash)

    if pr_num and pr_num in seen_prs:
      continue
    if pr_num:
      seen_prs.add(pr_num)

    classification = classify_commit(subject, body)
    if is_excluded_commit(subject, body, classification["scope"]):
      continue

    pr_link = f"[PR #{pr_num}]({REPO_URL}/pull/{pr_num})" if pr_num else f"`{commit_hash[:7]}`"

    commits.append(
        {
            "hash": commit_hash[:9],
            "author": author,
            "date": date,
            "raw_subject": subject,
            "summary": classification["clean_subject"],
            "pr_number": pr_num,
            "pr_link": pr_link,
            "section": classification["section"],
            "category": classification["category"],
            "body": body,
        }
    )

  return commits


def group_multipart_commits(commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Consolidates multi-part PR series into a single entry."""
  grouped: List[Dict[str, Any]] = []
  series_map: Dict[str, Dict[str, Any]] = {}

  # 1. Explicit numbered series: "[NNX] Delete Linen (5/5): details"
  series_paren_re = re.compile(r"^((?:\[[^\]]+\]\s*)?[^(:]+?)\s*\([^)]*\b\d+/\d+\)\s*:\s*(.+)$")
  series_bracket_re = re.compile(r"^\[([^\]]+?)\s+\d+/\d+\]\s*:?\s*(.+)$")

  # 2. Topic grouping rules for multi-PR features within the same section
  topic_rules = [
      ("Raiden Weight Sync", re.compile(r"\braiden\b", re.IGNORECASE), "Post-Training"),
      (
          "FP8 Quantization & Checkpointing",
          re.compile(r"\bfp8\b|dequantize-on-load", re.IGNORECASE),
          "Performance",
      ),
      (
          "DeepSeek-V4",
          re.compile(r"\bdeepseek[- ]?v?4\b|\bdsv4\b", re.IGNORECASE),
          "Pre-Training",
      ),
  ]

  for c in commits:
    summary = c["summary"].strip()
    m1 = series_paren_re.match(summary)
    m2 = series_bracket_re.match(summary) if not m1 else None

    prefix = None
    detail = None
    override_category = None

    if m1:
      prefix = m1.group(1).strip()
      detail = m1.group(2).strip()
    elif m2:
      prefix = m2.group(1).strip()
      detail = m2.group(2).strip()
    else:
      for topic_name, topic_re, target_cat in topic_rules:
        if topic_re.search(summary) and c["section"] == "Changes":
          prefix = topic_name
          # Strip redundant tags like [Memory], [DSV4], [Deepseek v4], PR #XXXX:
          cleaned = re.sub(r"^(?:\[[^\]]+\]\s*|PR\s*#\d+:\s*)+", "", summary, flags=re.IGNORECASE)
          cleaned = re.sub(r"\s*\[deepseek[- ]?v?4\]", "", cleaned, flags=re.IGNORECASE).strip()
          detail = cleaned
          override_category = target_cat
          break

    if not prefix or not detail:
      grouped.append(dict(c))
      continue

    series_key = f"{c['section']}::{prefix.lower()}"
    if series_key not in series_map:
      entry = dict(c)
      if override_category:
        entry["category"] = override_category
      entry["_series_prefix"] = prefix
      entry["_series_details"] = [detail]
      entry["_series_pr_links"] = [c["pr_link"]]
      series_map[series_key] = entry
      grouped.append(entry)
    else:
      existing = series_map[series_key]
      if detail not in existing["_series_details"]:
        existing["_series_details"].append(detail)
      if c["pr_link"] not in existing["_series_pr_links"]:
        existing["_series_pr_links"].append(c["pr_link"])

  for item in grouped:
    if "_series_prefix" in item:
      prefix = item.pop("_series_prefix")
      details = item.pop("_series_details")
      pr_links = item.pop("_series_pr_links")
      item["summary"] = f"**{prefix}**: {'; '.join(details)}"
      item["pr_link"] = ", ".join(pr_links)

  return grouped


def format_commit_bullet(item: Dict[str, Any], indent: str = "") -> str:
  """Formats a single changelog bullet line."""
  summary = item["summary"]
  if summary and summary[0].islower():
    summary = summary[0].upper() + summary[1:]
  return f"{indent}- {summary} ({item['pr_link']})."


def generate_changelog_block(major_commits: List[Dict[str, Any]]) -> str:
  """Generates formatted Markdown lines."""
  if not major_commits:
    return ""
  consolidated = group_multipart_commits(major_commits)
  block_lines = []
  for section in ["Changes", "Bug Fixes", "Deprecations"]:
    section_commits = [c for c in consolidated if c["section"] == section]
    if not section_commits:
      continue
    block_lines.append(f"#### {section}")
    block_lines.append("")

    if section == "Deprecations":
      block_lines.extend(format_commit_bullet(item) for item in section_commits)
      block_lines.append("")
      continue

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for c in section_commits:
      by_category.setdefault(c["category"], []).append(c)
    for category, items in by_category.items():
      block_lines.append(f"- **{category}**:")
      block_lines.extend(format_commit_bullet(item, indent="  ") for item in items)
      block_lines.append("")
  return "\n".join(block_lines).strip()


def update_changelog_files(new_block: str, head_commit: str, repo_dir: str = ".") -> None:
  """Inserts new major commits and updates `**Last Updated**: <commit_hash>` in docs/release_notes.md."""
  marker = "<!-- Add new unreleased changes below this line -->"
  updated_tag = f"**Last Updated**: {head_commit}"
  targets = [
      os.path.join(repo_dir, "docs", "release_notes.md"),
  ]
  for path in targets:
    if not os.path.exists(path):
      continue
    with open(path, "r", encoding="utf-8") as f:
      content = f.read()
    existing_updated_re = re.compile(r"\*{0,2}Last Updated\*{0,2}\s*:\s*\*{0,2}`?[0-9a-f]{7,40}`?", re.IGNORECASE)
    new_content = existing_updated_re.sub(updated_tag, content, count=1)
    if marker in new_content:
      replacement = f"{marker}\n\n{new_block}"
      new_content = new_content.replace(marker, replacement, 1)
    elif "## Unreleased" in new_content:
      if not existing_updated_re.search(content):
        new_content = new_content.replace("## Unreleased", f"## Unreleased\n\n{updated_tag}\n\n{new_block}", 1)
      else:
        new_content = new_content.replace(updated_tag, f"{updated_tag}\n\n{new_block}", 1)
    else:
      continue
    if new_content != content:
      with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main() -> None:
  parser = argparse.ArgumentParser(description="Gather git changes for changelog since last updated commit.")
  parser.add_argument(
      "--last-commit", type=str, default=None, help="Explicit last checked commit hash (<last_commit>..HEAD)"
  )
  parser.add_argument(
      "--update",
      action="store_true",
      help="Automatically insert new candidate major changes and **Last Updated**: <commit> into docs/release_notes.md",
  )
  args = parser.parse_args()

  head_commit = get_head_commit()
  last_commit = args.last_commit or get_last_updated_commit()
  if last_commit is None:
    return
  range_label = f"{last_commit}..{head_commit}"

  raw_log = run_git_log(last_commit=last_commit)
  commits = parse_commits(raw_log)
  if not commits:
    print(f"No new commits ({range_label})")
    return

  new_changelog = generate_changelog_block(commits)
  if not new_changelog:
    print(f"No new changelog ({range_label})")
    return

  if args.update:
    update_changelog_files(new_changelog, head_commit)
  else:
    print(new_changelog)


if __name__ == "__main__":
  main()
