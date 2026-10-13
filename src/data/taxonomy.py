"""Ask Ubuntu tag-based feasibility analysis for ResolveX's six issue categories.

Tags are weak-label evidence, not classifier features or verified ground truth.
"""

from collections import Counter, defaultdict
from itertools import combinations
import re

TAXONOMY = {
    "Network & Connectivity": {"networking", "wireless", "wifi", "internet", "network-manager", "ethernet", "dns", "vpn", "bluetooth", "connection", "modem", "broadband", "ip", "proxy"},
    "Boot & Startup": {"boot", "grub2", "grub", "uefi", "dual-boot", "boot-repair", "startup", "initramfs", "live-usb", "systemd", "recovery-mode", "bios"},
    "Software & Packages": {"apt", "package-management", "software-installation", "updates", "upgrade", "ppa", "dpkg", "snap", "repository", "dependencies", "apt-get", "software-center", "installation"},
    "Hardware & Drivers": {"drivers", "nvidia", "graphics", "intel", "amd-graphics", "sound", "audio", "touchpad", "keyboard", "mouse", "webcam", "printer", "usb", "display", "monitor", "gpu"},
    "Storage & Filesystems": {"partitioning", "hard-drive", "filesystem", "mount", "disk", "ssd", "usb-drive", "external-hdd", "ntfs", "ext4", "data-recovery", "lvm", "swap", "disk-usage"},
    "Access & Permissions": {"permissions", "sudo", "root", "users", "user-management", "password", "authentication", "login", "file-permissions", "ownership", "security"},
}
TAG_MARKUP = re.compile(r"<([^<>]+)>")


def parse_tags(raw):
    """Read both |tag|tag| derived CSVs and <tag><tag> dump markup."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    raw = raw.strip()
    if "|" in raw:
        return [tag.strip() for tag in raw.split("|") if tag.strip()]
    if raw.startswith("<"):
        return TAG_MARKUP.findall(raw)
    return []  # Unexpected formats must not silently become one fabricated tag.


def match_categories(tags):
    tag_set = set(tags)
    return [name for name, selected in TAXONOMY.items() if tag_set & selected]


def audit_questions(rows):
    """Stream CSV dict rows; return aggregates without retaining question bodies."""
    summary = Counter()
    tag_counts = Counter()
    any_match = Counter()
    exclusive = Counter()
    accepted_by_category = Counter()
    answered_by_category = Counter()
    selected_tag_counts = defaultdict(Counter)
    overlap_patterns = Counter()
    pairwise_overlap = Counter()
    unknown_formats = Counter()

    for row in rows:
        summary["total"] += 1
        raw_tags = row.get("Tags", "") or ""
        tags = parse_tags(raw_tags)
        if raw_tags.strip() and not tags:
            unknown_formats[raw_tags[:80]] += 1
        tag_set = set(tags)
        tag_counts.update(tag_set)
        accepted = bool(str(row.get("AcceptedAnswerId", "") or "").strip())
        try:
            answered = int(row.get("AnswerCount", 0) or 0) > 0
        except (ValueError, TypeError):
            answered = False
        summary["accepted"] += accepted
        summary["answered"] += answered
        matched = match_categories(tag_set)
        if not matched:
            summary["unmatched"] += 1
            continue
        for category in matched:
            any_match[category] += 1
            accepted_by_category[category] += accepted
            answered_by_category[category] += answered
            selected_tag_counts[category].update(tag_set & TAXONOMY[category])
        if len(matched) == 1:
            summary["single"] += 1
            exclusive[matched[0]] += 1
        else:
            summary["multi"] += 1
            overlap_patterns[" + ".join(sorted(matched))] += 1
            pairwise_overlap.update(combinations(sorted(matched), 2))

    return {
        "summary": summary,
        "global_tags": tag_counts,
        "any_match": any_match,
        "exclusive": exclusive,
        "accepted_by_category": accepted_by_category,
        "answered_by_category": answered_by_category,
        "selected_tag_counts": selected_tag_counts,
        "overlap_patterns": overlap_patterns,
        "pairwise_overlap": pairwise_overlap,
        "unknown_tag_formats": unknown_formats,
    }
