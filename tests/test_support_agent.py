import unittest
from unittest.mock import Mock

import numpy as np

from src.support_agent import (
    SupportAgent,
    _add_documented_dns_query_passage,
    _missing_package_repair_details,
    _package_manager_error_status,
    _assess_apt_transaction,
    _assess_recommendation_safety,
    _answer_citations_cover_claims,
    _has_citation_action_anchor,
    _final_status,
    _contains_executable_command,
    _unsafe_instruction,
)


# Verbatim diagnostic lines from Ask Ubuntu question 1503371, used by the
# preserved TEST1-RUN-001 evaluation. Keep this fixture local to the test.
ACTUAL_APT_DPKG_LOG = """Remove linux-image-6.5.0-14-generic (6.5.0-14.14~22.04.1) ...
dpkg: linux-image-6.5.0-14-generic package processing error (--remove) :
 the installed linux-image-6.5.0-14-generic script post-removal package sub-process returned an output status of error 1
E: Sub-process /usr/bin/dpkg returned an error code (1)"""


class _Pipeline:
    classes_ = np.asarray(["Network & Connectivity", "Software & Packages"])

    def decision_function(self, _texts):
        return np.asarray([[3.0, 0.2]])


def _artifact():
    return {"pipeline": _Pipeline(), "temperature": 1.0, "review_policy": {"threshold": 0.6}}


def _official_hit(**updates):
    return {
        "title": "Ubuntu 24.04 resolvectl reference",
        "url": "https://manpages.ubuntu.com/noble/resolvectl.1.html",
        "text": "resolvectl query example.com\nquery HOSTNAME\nResolve domain names and report the result.",
        "release_scope": "Ubuntu 24.04 LTS",
        "source_type": "official_ubuntu_manpage",
        "supported_releases": ["24.04"],
        "scope_status": "match",
        "release_scope_status": "match",
        "product_scope_status": "match",
        "categories": ["Network & Connectivity"],
        "score": 0.9,
        **updates,
    }


class _OfficialRetriever:
    chunks = [object()]

    def __init__(self, hits=None):
        self.hits = hits or [_official_hit()]
        self.search_calls = []

    def search(self, _query, **kwargs):
        self.search_calls.append(kwargs)
        return self.hits

    @staticmethod
    def _scope_metadata(_record, product_type=None, ubuntu_release=None):
        return {
            "scope_status": "match",
            "product_scope_status": "match" if product_type else "unknown",
            "release_scope_status": "match" if ubuntu_release else "unknown",
        }


class _CommunityRetriever:
    def __init__(self, weak=False):
        self.weak = weak
        self.calls = []

    def retrieve(self, title, body, k, exclude_question_ids=None):
        self.calls.append((title, body, k, exclude_question_ids))
        return {"evidence": [], "weak_evidence": self.weak, "method": "hybrid"}


def _agent(draft=None, *, official=None, community=None):
    chain = Mock()
    chain.invoke.return_value = {"parsed": draft or {
        "status": "recommendation",
        "summary": "Check whether the host name resolves.",
        "steps": [{
            "instruction": "Check domain names.",
            "command": "resolvectl query example.com",
            "citations": [{
                "source_id": "OD-1",
                "quote": "resolvectl query example.com\nquery HOSTNAME\nResolve domain names",
            }],
        }],
        "missing_information": [],
        "review_reasons": [],
    }}
    return SupportAgent(_artifact(), community or _CommunityRetriever(), chain,
                        official or _OfficialRetriever()), chain


class SupportAgentTests(unittest.TestCase):
    def test_read_only_path_display_is_allowed_but_other_variables_stay_blocked(self):
        self.assertFalse(_unsafe_instruction("Check PATH by running `echo $PATH`."))
        self.assertTrue(_unsafe_instruction("Run `echo $PACKAGE`."))
        self.assertTrue(_unsafe_instruction("Run `sudo echo $PATH`."))

    def test_dns_query_syntax_passage_is_added_from_release_matched_official_page(self):
        retriever = _OfficialRetriever([{
            **_official_hit(), "url": "https://ubuntu.test/resolvectl",
            "source_type": "official_ubuntu_manpage", "release_scope_status": "match",
        }])
        retriever.chunks = [{
            "url": "https://ubuntu.test/resolvectl", "title": "resolvectl",
            "text": "query HOSTNAME|ADDRESS...\nResolve domain names.", "part": 2,
            "source_type": "official_ubuntu_manpage",
            "release_scope": "Ubuntu 24.04 LTS", "supported_releases": ["24.04"],
        }]
        result = _add_documented_dns_query_passage(
            retriever.search("DNS", product_type="ubuntu_server", ubuntu_release="24.04"),
            retriever, "Temporary failure resolving host", "ubuntu_server", "24.04",
        )
        self.assertEqual(len(result), 2)
        self.assertIn("query HOSTNAME", result[1]["text"])
        self.assertEqual(result[1]["scope_status"], "match")
        self.assertEqual(result[1]["score"], 0.9)

    def test_prepare_exposes_classification_and_each_retrieval_lane(self):
        community = _CommunityRetriever()
        agent, _ = _agent(community=community)
        prepared = agent.prepare("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")

        self.assertEqual(prepared["category"], "Network & Connectivity")
        self.assertEqual(prepared["retrieval_lanes"]["askubuntu"]["status"], "empty")
        self.assertEqual(prepared["retrieval_lanes"]["official_ubuntu_docs"]["status"], "ok")
        self.assertEqual(community.calls[0][2], 5)

    def test_generation_error_is_reported_without_leaking_credentials(self):
        agent, chain = _agent()
        chain.invoke.side_effect = RuntimeError(
            "provider failed GEMINI_API_KEY=AIzaSyAbcdefghijklmnopqrstuvwxyz0123456789 "
            "Authorization: Bearer shortsecret https://api.example/path?key=url-secret"
        )

        result = agent.run("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")

        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(result["error"].startswith("RuntimeError:"))
        self.assertIn(result["error"], result["review_reasons"][-1])
        self.assertLessEqual(len(result["error"]), 300)
        for secret in (
            "AIzaSyAbcdefghijklmnopqrstuvwxyz0123456789",
            "shortsecret",
            "url-secret",
        ):
            self.assertNotIn(secret, result["error"])

    def test_structured_output_parsing_error_is_surfaced_and_sanitized(self):
        agent, chain = _agent()
        chain.invoke.return_value = {
            "parsed": None,
            "parsing_error": ValueError(
                "invalid JSON for token=parser-secret "
                "https://api.example/parse?access_token=query-secret"
            ),
        }

        result = agent.run("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")

        self.assertTrue(result["error"].startswith("ValueError:"))
        self.assertIn("invalid JSON", result["error"])
        self.assertNotIn("parser-secret", result["error"])
        self.assertNotIn("query-secret", result["error"])
        self.assertIn(result["error"], result["review_reasons"][-1])

    def test_prepare_searches_full_unique_official_pool_before_scope_filtering(self):
        official = _OfficialRetriever([
            _official_hit(scope_status="mismatch", release_scope_status="mismatch"),
            _official_hit(title="Ubuntu 24.04 apt-get reference", scope_status="match"),
        ])
        official.chunks = [object() for _ in range(12)]
        agent, _ = _agent(official=official)

        prepared = agent.prepare("Package question", "Ubuntu 24.04", ubuntu_release="24.04")

        self.assertEqual(official.search_calls[0]["k"], 12)
        self.assertTrue(official.search_calls[0]["deduplicate_pages"])
        self.assertEqual(len(prepared["sources"]), 1)
        self.assertEqual(prepared["retrieval_lanes"]["official_ubuntu_docs"]["excluded_scope_mismatch"], ["OD-1"])

    def test_prepare_passes_exclusions_into_retrieval(self):
        community = _CommunityRetriever()
        agent, _ = _agent(community=community)

        agent.prepare("Question", "Body", exclude_question_ids=["1503371"])

        self.assertEqual(community.calls[0][3], ["1503371"])

    def test_missing_package_repair_details_skip_generation(self):
        agent, chain = _agent()
        result = agent.run("apt upgrade has unmet dependencies")

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["missing_information"], [
            "Ubuntu version", "complete package-manager error message",
        ])
        chain.invoke.assert_not_called()

    def test_broken_package_dependencies_phrase_also_requests_context(self):
        agent, chain = _agent()
        result = agent.run("Broken package dependencies during apt upgrade")

        self.assertEqual(result["status"], "needs_review")
        self.assertIn("Ubuntu version", result["missing_information"])
        chain.invoke.assert_not_called()

    def test_out_of_scope_ticket_is_not_sent_to_model(self):
        agent, chain = _agent()
        result = agent.run("Windows Server domain controller", "Active Directory replication failed")

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["steps"], [])
        chain.invoke.assert_not_called()

    def test_historical_windows_mention_does_not_mark_ubuntu_ticket_out_of_scope(self):
        agent, _ = _agent()
        title = "Can I chose where an app is installed using apt?"
        body = (
            "I just switch from Windows to Ubuntu. I have 2 drives, one is a HDD and another is an SSD. "
            "I use my SSD for the home directory, documents, and apps I use all the time. "
            "I use the hard drive for programs and games. I have steam and I can chose where steam "
            "installs games. But when I download non games or games without steam I don't get an "
            "option to chose where to download them. I used to be able when I ran the MSI or EXE "
            "to chose where to download the app. In linux it's different where you use apt. "
            "How do I chose where to download stuff?"
        )

        prepared = agent.prepare(title, body)

        self.assertFalse(prepared["out_of_scope"])

    def test_ubuntu_server_active_directory_integration_is_in_scope(self):
        agent, _ = _agent()
        prepared = agent.prepare(
            "Ubuntu Server 22.04 joined to a Windows Active Directory domain",
            "The Ubuntu server uses realmd and sssd to join the Windows domain.",
        )

        self.assertFalse(prepared["out_of_scope"])

    def test_verified_step_keeps_source_and_engineer_review(self):
        agent, _ = _agent()
        result = agent.run(
            "DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop"
        )

        self.assertEqual(result["status"], "recommendation")
        self.assertTrue(result["engineer_review_required"])
        self.assertEqual(result["citation_check"], "passed")
        self.assertEqual(result["answer_validation"]["status"], "not_provided")
        self.assertEqual(result["step_validation"]["status"], "passed")
        self.assertEqual(result["steps"][0]["citations"][0]["source_id"], "OD-1")
        self.assertEqual(result["steps"][0]["command"], "resolvectl query example.com")

    def test_executable_step_without_command_is_withheld(self):
        draft = {
            "status": "recommendation", "summary": "Install the requested package.",
            "steps": [{
                "instruction": "Run the package installation command.",
                "citations": [{"source_id": "OD-1", "quote": "Resolve domain names"}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run("Install an Ubuntu package", "Ubuntu 24.04")

        self.assertEqual(result["steps"], [])
        self.assertEqual(len(result["withheld_steps"]), 1)
        self.assertTrue(any("without a separate command field" in issue
                            for issue in result["withheld_steps"][0]["issues"]))

    def test_non_executable_information_request_may_omit_command(self):
        quote = "Ubuntu 24.04 LTS"
        draft = {
            "status": "needs_review", "summary": "Collect the missing release.",
            "steps": [{
                "instruction": "Provide the Ubuntu release in use.",
                "citations": [{"source_id": "OD-1", "quote": quote}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)
        official = _OfficialRetriever([_official_hit(text=quote)])
        agent.official_retriever = official

        result = agent.run("Package question", "Ubuntu 24.04", product_type="ubuntu_desktop")

        self.assertEqual(result["step_validation"]["status"], "passed")
        self.assertIsNone(result["steps"][0]["command"])

    def test_command_not_present_in_cited_quote_is_withheld(self):
        draft = {
            "status": "recommendation", "summary": "Install the requested package.",
            "steps": [{
                "instruction": "Install the requested package.",
                "command": "apt-get install vim",
                "citations": [{"source_id": "OD-1", "quote": "query HOSTNAME"}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run("Install an Ubuntu package", "Ubuntu 24.04")

        self.assertEqual(result["steps"], [])
        self.assertTrue(any("exact command is not present" in issue
                            for issue in result["withheld_steps"][0]["issues"]))

    def test_dns_query_command_is_built_from_ticket_and_scoped_quote(self):
        quote = "query HOSTNAME\nResolve domain names and report the result."
        draft = {
            "status": "recommendation", "summary": "Check DNS resolution.",
            "steps": [{
                "instruction": "Query the hostname to check whether it resolves.",
                "citations": [{"source_id": "OD-1", "quote": quote}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run(
            "Ubuntu Server 24.04 cannot resolve www.google.com",
            "DNS lookups for www.google.com time out.",
            product_type="ubuntu_server", ubuntu_release="24.04",
        )

        self.assertEqual(result["step_validation"]["status"], "passed")
        self.assertEqual(result["steps"][0]["command"], "resolvectl query www.google.com")

    def test_resolvectl_service_quote_does_not_support_hostname_query(self):
        quote = "resolvectl service _xmpp-server._tcp gmail.com"
        draft = {
            "status": "recommendation", "summary": "Check DNS resolution.",
            "steps": [{
                "instruction": "Run a read-only query for the hostname using resolvectl.",
                "citations": [{"source_id": "OD-1", "quote": quote}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        official = _OfficialRetriever([_official_hit(text=quote)])
        agent, _ = _agent(draft, official=official)

        result = agent.run(
            "Ubuntu Server 24.04 cannot resolve www.google.com",
            "DNS lookups for www.google.com time out.",
            product_type="ubuntu_server", ubuntu_release="24.04",
        )

        self.assertEqual(result["steps"], [])
        self.assertEqual(result["step_validation"]["status"], "failed")
        self.assertEqual(result["withheld_steps"][0].get("command"), None)

    def test_step_command_safety_checks_the_command_field(self):
        command = "sudo apt-get --fix-broken install"
        draft = {
            "status": "recommendation", "summary": "Review the package operation.",
            "steps": [{
                "instruction": "Review the package repair operation.",
                "command": command,
                "citations": [{"source_id": "OD-1", "quote": command}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft, official=_OfficialRetriever([_official_hit(text=command)]))

        result = agent.run("Repair Ubuntu packages", "Ubuntu 24.04")

        self.assertEqual(result["steps"], [])
        self.assertEqual(result["safety_assessment"]["decision"], "escalate")

    def test_information_only_answer_can_be_returned_with_verified_citation(self):
        quote = (
            "resolvectl query example.com\nquery HOSTNAME\n"
            "Resolve domain names and report the result."
        )
        draft = {
            "status": "recommendation",
            "summary": "A sourced explanation is available.",
            "answer": "The documented query operation resolves the supplied host name and reports the result.",
            "answer_citations": [{"source_id": "OD-1", "quote": quote}],
            "steps": [],
            "missing_information": [],
            "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run(
            "What does the documented hostname query do?", "Ubuntu 24.04",
            product_type="ubuntu_desktop",
        )

        self.assertEqual(result["answer"], draft["answer"])
        self.assertEqual(result["answer_citations"][0]["source_id"], "OD-1")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["citation_check"], "passed")
        self.assertEqual(result["answer_validation"]["status"], "passed")
        self.assertEqual(result["step_validation"]["status"], "no_steps")

    def test_missing_information_keeps_a_general_answer_as_review_draft(self):
        quote = "query HOSTNAME\nResolve domain names and report the result."
        draft = {
            "status": "recommendation",
            "summary": "The general explanation is available; collect the missing detail before advising further.",
            "answer": "The documented query operation resolves a host name and reports the result.",
            "answer_citations": [{"source_id": "OD-1", "quote": quote}],
            "steps": [],
            "missing_information": ["Ubuntu release"],
            "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run(
            "What does the documented hostname query do?", "Ubuntu release unknown",
            product_type="ubuntu_desktop",
        )

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["answer"], draft["answer"])
        self.assertEqual(result["missing_information"], ["Ubuntu release"])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["answer_validation"]["status"], "passed")

    def test_invalid_answer_citation_withholds_answer_and_factual_summary(self):
        draft = {
            "status": "recommendation",
            "summary": "APT installs programs in system directories.",
            "answer": "APT packages place files in system-managed locations.",
            "answer_citations": [{
                "source_id": "OD-MISSING",
                "quote": "APT installs packages in system directories.",
            }],
            "steps": [],
            "missing_information": [],
            "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run("Where does apt install packages?", "Ubuntu 24.04")

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["answer_citations"], [])
        self.assertEqual(
            result["summary"],
            "The explanation was withheld because its evidence could not be verified.",
        )
        self.assertEqual(result["citation_check"], "failed")
        self.assertEqual(result["answer_validation"]["status"], "failed")
        self.assertEqual(result["step_validation"]["status"], "no_steps")
        self.assertTrue(any(
            "unknown source ID 'OD-MISSING'" in reason
            for reason in result["review_reasons"]
        ))

    def test_answer_quote_mismatch_reports_the_source_id(self):
        draft = {
            "status": "recommendation",
            "summary": "A factual summary that must be withheld.",
            "answer": "The answer makes a claim about package locations.",
            "answer_citations": [{
                "source_id": "OD-1",
                "quote": "This wording is absent from the source.",
            }],
            "steps": [],
            "missing_information": [],
            "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run("Where does apt install packages?", "Ubuntu 24.04")

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["answer_validation"]["status"], "failed")
        detail = result["answer_validation"]["issues_detail"][0]
        self.assertEqual(detail["raw_quote"], "This wording is absent from the source.")
        self.assertEqual(detail["source_id"], "OD-1")
        self.assertEqual(
            detail["normalized_quote"], "this wording is absent from the source."
        )
        self.assertGreater(detail["normalized_source_excerpt_length"], 0)
        self.assertEqual(
            result["summary"],
            "The explanation was withheld because its evidence could not be verified.",
        )
        self.assertTrue(any(
            "quote does not match source 'OD-1'" in reason
            for reason in result["review_reasons"]
        ))

    def test_topically_weak_answer_is_retained_only_as_engineer_review_draft(self):
        quote = "query HOSTNAME\nResolve domain names and report the result."
        draft = {
            "status": "recommendation",
            "summary": "A draft explanation needs review.",
            "answer": "This unrelated topic reports a DNS status.",
            "answer_citations": [{"source_id": "OD-1", "quote": quote}],
            "steps": [],
            "missing_information": [],
            "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run(
            "What does the documented hostname query do?", "Ubuntu 24.04",
            product_type="ubuntu_desktop",
        )

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["answer"], draft["answer"])
        self.assertEqual(result["citation_check"], "failed")
        self.assertEqual(result["answer_validation"]["status"], "needs_review")
        self.assertEqual(result["step_validation"]["status"], "no_steps")
        self.assertTrue(any("lexical support" in reason for reason in result["review_reasons"]))

    def test_unsafe_step_is_withheld_without_discarding_valid_information_answer(self):
        quote = (
            "resolvectl query example.com\nquery HOSTNAME\n"
            "Resolve domain names and report the result."
        )
        draft = {
            "status": "recommendation",
            "summary": "Review the sourced explanation.",
            "answer": "The documented query operation resolves the supplied host name and reports the result.",
            "answer_citations": [{"source_id": "OD-1", "quote": quote}],
            "steps": [{
                "instruction": "Review the package repair operation.",
                "command": "sudo apt --fix-broken install",
                "citations": [{"source_id": "OD-1", "quote": "package handling"}],
            }, {
                "instruction": "Check whether the hostname resolves.",
                "command": "resolvectl query example.com",
                "citations": [{"source_id": "OD-1", "quote": quote}],
            }],
            "missing_information": [],
            "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run(
            "What does the documented hostname query do?", "Ubuntu 24.04",
            product_type="ubuntu_desktop",
        )

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["answer"], draft["answer"])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["answer_validation"]["status"], "passed")
        self.assertEqual(result["step_validation"]["status"], "failed")
        self.assertTrue(result["step_validation"]["issues"])
        self.assertEqual(len(result["withheld_steps"]), 2)
        self.assertEqual(
            result["withheld_steps"][0]["instruction"],
            "Review the package repair operation.",
        )
        self.assertTrue(result["withheld_steps"][0]["audit_only"])
        self.assertTrue(result["withheld_steps"][0]["issues"])
        self.assertIn("otherwise-passing", result["withheld_steps"][1]["issues"][0])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["safety_assessment"]["decision"], "escalate")
        self.assertTrue(any("Safety escalation" in reason for reason in result["review_reasons"]))

    def test_mixed_sources_are_returned_official_first(self):
        community = _CommunityRetriever()
        community.retrieve = lambda *_args, **_kwargs: {
            "evidence": [{
                "question_id": "100", "title": "Community case",
                "answer_text": "An example answer.", "rank": 1, "score": 0.7,
                "sources": {"answer": {"source_url": "https://askubuntu.com/a/101",
                                        "creation_date": "2024-01-01T00:00:00Z"}},
            }],
            "weak_evidence": False, "method": "dense",
        }
        agent, _ = _agent(community=community)
        prepared = agent.prepare("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")
        self.assertEqual(prepared["sources"][0]["lane"], "official_ubuntu_docs")
        self.assertEqual(prepared["sources"][1]["lane"], "askubuntu")

    def test_official_candidates_prefer_release_match_then_confident_category_alignment(self):
        official = _OfficialRetriever([
            _official_hit(
                title="Hardware reference", categories=["Hardware & Drivers"],
                rank=1, score=0.95,
            ),
            _official_hit(
                title="Package reference", categories=["Software & Packages"],
                rank=2, score=0.8,
            ),
        ])
        agent, _ = _agent(official=official)

        prepared = agent.prepare(
            "Install a software package on Ubuntu 24.04", "APT package question",
            category_override="Software & Packages",
        )

        self.assertEqual(prepared["retrieval_lanes"]["official_ubuntu_docs"]["preferred_category"], "Software & Packages")
        self.assertEqual(prepared["sources"][0]["title"], "Package reference")
        self.assertEqual(prepared["sources"][0]["rank"], 2)
        self.assertEqual(prepared["sources"][0]["selection_rank"], 1)

    def test_official_docs_drop_low_relative_score_and_record_count(self):
        official = _OfficialRetriever([
            _official_hit(title="Relevant", score=0.8, rank=1),
            _official_hit(title="Weak", score=0.41, rank=2),
            _official_hit(title="Very weak", score=0.19, rank=3),
        ])
        agent, _ = _agent(official=official)

        prepared = agent.prepare("Ubuntu 24.04 DNS lookup", "Need a DNS check")

        lane = prepared["retrieval_lanes"]["official_ubuntu_docs"]
        self.assertEqual([source["title"] for source in prepared["sources"]], ["Relevant", "Weak"])
        self.assertEqual(lane["dropped_low_relevance"], 1)

    def test_official_docs_always_keep_top_result_below_absolute_floor(self):
        official = _OfficialRetriever([_official_hit(score=0.12, rank=1)])
        agent, _ = _agent(official=official)

        prepared = agent.prepare("Ubuntu 24.04 DNS lookup", "Need a DNS check")

        lane = prepared["retrieval_lanes"]["official_ubuntu_docs"]
        self.assertEqual(len(prepared["sources"]), 1)
        self.assertEqual(prepared["sources"][0]["score"], 0.12)
        self.assertEqual(lane["dropped_low_relevance"], 0)

    def test_official_docs_without_scores_keep_only_top_candidate(self):
        official = _OfficialRetriever([
            _official_hit(title="First candidate", score=None, rank=1),
            _official_hit(title="Unscored candidate", score=None, rank=2),
        ])
        agent, _ = _agent(official=official)

        prepared = agent.prepare("Ubuntu 24.04 DNS lookup", "Need a DNS check")

        self.assertEqual([source["title"] for source in prepared["sources"]], ["First candidate"])
        self.assertEqual(
            prepared["retrieval_lanes"]["official_ubuntu_docs"]["dropped_low_relevance"],
            1,
        )

    def test_answer_relevance_screen_checks_each_claim_not_one_shared_word(self):
        citations = [{
            "source_id": "OD-1",
            "quote": "APT package manager installs package files in standard locations.",
        }]

        self.assertTrue(_answer_citations_cover_claims(
            "The package manager installs package files.", citations,
        ))
        self.assertFalse(_answer_citations_cover_claims(
            "The package manager installs package files. Steam games store library data separately.",
            citations,
        ))

    def test_uncited_community_hits_do_not_downgrade_official_recommendation(self):
        community = _CommunityRetriever()
        community.retrieve = lambda *_args, **_kwargs: {
            "evidence": [{
                "question_id": "100", "title": "Old unrelated community case",
                "answer_text": "Unrelated answer.", "rank": 1, "score": 0.7,
                "sources": {"answer": {"source_url": "https://askubuntu.com/a/101",
                                        "creation_date": "2010-01-01T00:00:00Z"}},
            }],
            "weak_evidence": True, "method": "dense",
        }
        agent, _ = _agent(community=community)
        result = agent.run("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")

        self.assertEqual(result["status"], "recommendation")
        self.assertEqual([source["source_id"] for source in result["sources"]], ["OD-1"])
        self.assertEqual(len(result["retrieved_sources"]), 2)
        self.assertFalse(any("Ask Ubuntu" in reason for reason in result["review_reasons"]))

    def test_quote_not_found_withholds_all_draft_steps(self):
        draft = {
            "status": "recommendation", "summary": "Try this.",
            "steps": [{"instruction": "Run a check.", "command": "resolvectl query example.com", "citations": [
                {"source_id": "OD-1", "quote": "This quote is not in the source."},
            ]}], "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)
        result = agent.run("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["citation_check"], "failed")
        self.assertEqual(result["step_validation"]["status"], "failed")

    def test_unsafe_command_withholds_all_draft_steps(self):
        draft = {
            "status": "recommendation", "summary": "Try this.",
            "steps": [{"instruction": "Change permissions recursively.", "command": "sudo chmod -R 777 /", "citations": [
                {"source_id": "OD-1", "quote": "query HOSTNAME"},
            ]}], "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)
        result = agent.run("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["steps"], [])
        self.assertIn("unsafe", result["review_reasons"][0])

    def test_release_mismatch_is_filtered_before_generation(self):
        official = _OfficialRetriever([_official_hit(
            scope_status="mismatch", release_scope_status="mismatch",
        )])
        agent, chain = _agent(official=official)
        prepared = agent.prepare("DNS lookup fails", "Ubuntu 24.04", product_type="ubuntu_desktop")
        self.assertEqual(prepared["sources"], [])
        self.assertEqual(prepared["retrieval_lanes"]["official_ubuntu_docs"]["excluded_scope_mismatch"], ["OD-1"])
        result = agent.generate(prepared)
        self.assertEqual(result["status"], "needs_review")
        chain.invoke.assert_not_called()

    def test_invalid_ticket_rejected(self):
        agent, _ = _agent()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            agent.run("", "")


class PackageErrorDetectionTests(unittest.TestCase):
    def test_actual_ticket_apt_and_dpkg_logs_are_present_and_sufficient(self):
        status = _package_manager_error_status(ACTUAL_APT_DPKG_LOG)

        self.assertEqual(status, {
            "error_log_present": True,
            "error_details_sufficient": True,
        })

    def test_apt_e_prefix_alone_is_present_but_not_sufficient(self):
        status = _package_manager_error_status("E:")

        self.assertTrue(status["error_log_present"])
        self.assertFalse(status["error_details_sufficient"])

    def test_dpkg_prefix_alone_is_present_but_not_sufficient(self):
        status = _package_manager_error_status("dpkg:")

        self.assertTrue(status["error_log_present"])
        self.assertFalse(status["error_details_sufficient"])

    def test_detailed_apt_error_is_sufficient(self):
        status = _package_manager_error_status(
            "E: Unable to locate package zoom-meeting"
        )

        self.assertTrue(status["error_log_present"])
        self.assertTrue(status["error_details_sufficient"])

    def test_detailed_dpkg_error_is_sufficient(self):
        status = _package_manager_error_status(
            "dpkg: linux-image-6.5.0-14-generic package processing error (--remove)"
        )

        self.assertTrue(status["error_log_present"])
        self.assertTrue(status["error_details_sufficient"])

    def test_no_package_manager_error_log_is_not_present_or_sufficient(self):
        status = _package_manager_error_status("apt upgrade has unmet dependencies")

        self.assertEqual(status, {
            "error_log_present": False,
            "error_details_sufficient": False,
        })

    def test_repair_ticket_keeps_missing_information_check_for_incomplete_log(self):
        missing = _missing_package_repair_details(
            "Ubuntu 22.04 apt upgrade has unmet dependencies. E: error"
        )

        self.assertEqual(missing, ["complete package-manager error message"])

    def test_repair_ticket_with_real_logs_does_not_request_error_message(self):
        missing = _missing_package_repair_details(
            "Ubuntu 22.04 broken package dependencies.\n" + ACTUAL_APT_DPKG_LOG
        )

        self.assertEqual(missing, [])

    def test_prepare_exposes_error_status_for_the_real_ticket(self):
        agent, _ = _agent()
        prepared = agent.prepare(
            "Broken package dependencies on Ubuntu 22.04",
            ACTUAL_APT_DPKG_LOG,
        )

        self.assertTrue(prepared["error_log_present"])
        self.assertTrue(prepared["error_details_sufficient"])
        self.assertEqual(prepared["missing_information"], [])

    def test_repair_ticket_without_release_still_requests_release(self):
        missing = _missing_package_repair_details(
            "Broken package dependencies.\n"
            "dpkg: zoom package processing error (--remove)\n"
            "E: Sub-process /usr/bin/dpkg returned an error code (1)"
        )

        self.assertEqual(missing, ["Ubuntu version"])

    def test_explicit_release_is_used_instead_of_kernel_package_version(self):
        text = (
            "Broken package dependencies.\n"
            "Remove linux-image-6.5.0-14-generic (6.5.0-14.14~22.04.1) ...\n"
            "dpkg: linux-image-6.5.0-14-generic package processing error (--remove)\n"
            "E: Sub-process /usr/bin/dpkg returned an error code (1)"
        )

        self.assertEqual(_missing_package_repair_details(text), ["Ubuntu version"])
        self.assertEqual(_missing_package_repair_details(text, ubuntu_release="22.04"), [])

    def test_non_repair_ticket_does_not_request_package_error_details(self):
        missing = _missing_package_repair_details(
            "Ubuntu 22.04: how do I list installed packages?"
        )

        self.assertEqual(missing, [])


class SafetyGateTests(unittest.TestCase):
    KERNEL_TRANSACTION = """The following packages will be REMOVED:
  linux-image-6.5.0-14-generic
0 updated, 2 newly installed, 1 to be removed and 26 not updated."""

    REAL_TICKET_APT_TRANSACTIONS = """The following packages will be REMOVED:
  linux-image-6.5.0-14-generic zoom*
0 updated, 0 newly installed, 2 to be removed and 26 not updated.

The following additional packages will be installed:
  libxcb-cursor0 libxcb-xtest0
The following packages will be REMOVED:
  linux-image-6.5.0-14-generic
The following NEW packages will be installed:
  libxcb-cursor0 libxcb-xtest0
0 updated, 2 newly installed, 1 to be removed and 26 not updated."""

    def test_real_ticket_apt_transactions_parse_removals_without_install_list_noise(self):
        assessment = _assess_apt_transaction(self.REAL_TICKET_APT_TRANSACTIONS)

        self.assertEqual(assessment["status"], "known")
        self.assertTrue(assessment["kernel_removal_planned"])
        # The kernel appears in both real transaction plans but is listed once;
        # zoom* is APT's marked package token, while newly installed dependencies
        # are not removals.
        self.assertEqual(assessment["removed_packages"], [
            "linux-image-6.5.0-14-generic", "zoom",
        ])
        self.assertNotIn("the", assessment["removed_packages"])
        self.assertNotIn("libxcb-cursor0", assessment["removed_packages"])

    def test_real_ticket_incomplete_removal_list_remains_unknown_and_keeps_kernel_risk(self):
        text = """The following packages will be REMOVED:
  linux-image-6.5.0-14-generic
0 updated, 0 newly installed, 2 to be removed and 26 not updated."""

        assessment = _assess_apt_transaction(text)

        self.assertEqual(assessment["status"], "unknown")
        self.assertTrue(assessment["kernel_removal_planned"])

    def test_kernel_removal_is_detected_in_apt_transaction_plan(self):
        assessment = _assess_apt_transaction(self.KERNEL_TRANSACTION)

        self.assertEqual(assessment["status"], "known")
        self.assertTrue(assessment["kernel_removal_planned"])
        self.assertEqual(assessment["removed_packages"], ["linux-image-6.5.0-14-generic"])

    def test_kernel_removal_is_found_in_later_apt_transaction(self):
        text = (
            "The following packages will be REMOVED:\n  harmless-package\n"
            "0 updated, 0 newly installed, 1 to be removed and 0 not updated.\n"
            "The following packages will be REMOVED:\n  linux-image-6.5.0-14-generic\n"
            "0 updated, 0 newly installed, 1 to be removed and 0 not updated."
        )

        assessment = _assess_apt_transaction(text)

        self.assertEqual(assessment["status"], "known")
        self.assertTrue(assessment["kernel_removal_planned"])
        self.assertEqual(len(assessment["removed_packages"]), 2)

    def test_no_transaction_plan_is_reported_as_absent(self):
        assessment = _assess_apt_transaction("Ubuntu 22.04 package repair ticket")

        self.assertEqual(assessment["status"], "absent")
        self.assertFalse(assessment["kernel_removal_planned"])

    def test_incomplete_transaction_plan_is_reported_as_unknown(self):
        assessment = _assess_apt_transaction(
            "The following packages will be REMOVED:\n  linux-image-6.5.0-14-generic"
        )

        self.assertEqual(assessment["status"], "unknown")
        self.assertTrue(assessment["kernel_removal_planned"])

    def test_fix_broken_with_kernel_removal_is_escalated(self):
        transaction = _assess_apt_transaction(self.KERNEL_TRANSACTION)
        result = _assess_recommendation_safety(
            "Run `sudo apt --fix-broken install`.", transaction,
        )

        self.assertEqual(result["decision"], "escalate")
        self.assertTrue(any("kernel" in reason.lower() for reason in result["reasons"]))

    def test_fix_broken_without_transaction_result_is_escalated(self):
        transaction = _assess_apt_transaction("Broken package dependencies on Ubuntu 22.04")
        result = _assess_recommendation_safety(
            "Run `sudo apt --fix-broken install`.", transaction,
        )

        self.assertEqual(result["decision"], "escalate")
        self.assertTrue(any("transaction" in reason.lower() for reason in result["reasons"]))

    def test_read_only_diagnostics_remain_allowed_with_kernel_removal_risk(self):
        transaction = _assess_apt_transaction(self.KERNEL_TRANSACTION)
        for instruction in (
            "Run `dpkg --audit` and share the output.",
            "Read `/etc/default/grub` with `cat /etc/default/grub`.",
        ):
            with self.subTest(instruction=instruction):
                result = _assess_recommendation_safety(instruction, transaction)
                self.assertEqual(result["decision"], "allow")

    def test_kernel_removal_command_is_escalated(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        result = _assess_recommendation_safety(
            "Run `sudo apt remove linux-image-6.5.0-14-generic`.", transaction,
        )

        self.assertEqual(result["decision"], "escalate")

    def test_dpkg_state_file_modification_is_escalated_but_read_is_allowed(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        modify = _assess_recommendation_safety(
            "Run `sudo rm /var/lib/dpkg/info/zoom.postrm`.", transaction,
        )
        inspect = _assess_recommendation_safety(
            "Read ` /var/lib/dpkg/status` with `cat /var/lib/dpkg/status`.", transaction,
        )

        self.assertEqual(modify["decision"], "escalate")
        self.assertEqual(inspect["decision"], "allow")

    def test_direct_dpkg_package_mutation_is_escalated(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        result = _assess_recommendation_safety("Run `dpkg --configure -a`.", transaction)

        self.assertEqual(result["decision"], "escalate")

    def test_grub_modification_is_escalated_but_read_only_inspection_is_allowed(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        modify = _assess_recommendation_safety(
            "Edit `/etc/default/grub` with `sudo nano /etc/default/grub`.", transaction,
        )
        inspect = _assess_recommendation_safety(
            "Inspect `/etc/default/grub` using `grep -n . /etc/default/grub`.", transaction,
        )

        self.assertEqual(modify["decision"], "escalate")
        self.assertEqual(inspect["decision"], "allow")

    def test_apt_simulation_is_read_only_even_with_kernel_risk(self):
        transaction = _assess_apt_transaction(self.KERNEL_TRANSACTION)
        result = _assess_recommendation_safety(
            "Preview with `sudo apt --simulate --fix-broken install`.", transaction,
        )

        self.assertEqual(result["decision"], "allow")

    def test_apt_yes_is_not_treated_as_read_only(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        for instruction in (
            "Run `sudo apt -y install example`.",
            "Run `sudo apt-get -y remove example`.",
        ):
            with self.subTest(instruction=instruction):
                result = _assess_recommendation_safety(instruction, transaction)
                self.assertEqual(result["decision"], "escalate")

    def test_citation_must_be_topically_anchored_to_recommendation(self):
        self.assertFalse(_has_citation_action_anchor(
            "Run `grep -n . /etc/default/grub` to inspect GRUB settings.",
            "APT installs and removes packages.",
        ))
        self.assertTrue(_has_citation_action_anchor(
            "Run `resolvectl query example.com` and share the result.",
            "query HOSTNAME and resolve domain names.",
        ))

    def test_uncertain_or_conflicting_draft_cannot_keep_recommendation_status(self):
        candidate_step = [{"instruction": "candidate for engineer review"}]
        for reason in (
            "At least one source has unknown release scope.",
            "The cited sources contain conflicting guidance.",
        ):
            with self.subTest(reason=reason):
                self.assertEqual(
                    _final_status("recommendation", candidate_step, [reason]),
                    "needs_review",
                )
        self.assertEqual(_final_status("recommendation", candidate_step, []), "recommendation")
        self.assertEqual(
            _final_status(
                "recommendation", [], [], "general explanation",
                missing_information=["Ubuntu release"],
            ),
            "needs_review",
        )
        self.assertEqual(_final_status("unsupported", candidate_step, []), "unsupported")

    def test_ambiguous_privileged_command_is_escalated(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        result = _assess_recommendation_safety(
            "Run `sudo /usr/local/bin/package-repair --force`.", transaction,
        )

        self.assertEqual(result["decision"], "escalate")

    def test_incomplete_transaction_escalates_mutating_apt_action(self):
        transaction = _assess_apt_transaction(
            "The following packages will be REMOVED:\n  unknown-package"
        )
        result = _assess_recommendation_safety("Run `sudo apt install app`.", transaction)

        self.assertEqual(transaction["status"], "unknown")
        self.assertEqual(result["decision"], "escalate")

    def test_missing_transaction_result_escalates_apt_install(self):
        transaction = _assess_apt_transaction("Ubuntu 24.04 software installation request")
        result = _assess_recommendation_safety("Run `sudo apt install evince`.", transaction)

        self.assertEqual(transaction["status"], "absent")
        self.assertEqual(result["decision"], "escalate")

    def test_read_only_command_chained_with_mutation_is_escalated(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        result = _assess_recommendation_safety(
            "Run `cat /etc/default/grub && sudo rm /var/lib/dpkg/info/zoom.postrm`.",
            transaction,
        )

        self.assertEqual(result["decision"], "escalate")

    def test_mutating_command_outside_code_span_is_not_ignored(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        result = _assess_recommendation_safety(
            "Run `dpkg --audit`, then run sudo apt install example.", transaction,
        )

        self.assertEqual(result["decision"], "escalate")

    def test_unprivileged_file_mutations_are_escalated(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        for instruction in (
            "Run `rm ~/.config/example.conf`.",
            "Run `sed -i s/old/new/ /etc/example.conf`.",
            "Run `chmod 644 /etc/example.conf`.",
            "Run `mount /dev/sdb1 /mnt/data`.",
        ):
            with self.subTest(instruction=instruction):
                result = _assess_recommendation_safety(instruction, transaction)
                self.assertEqual(result["decision"], "escalate")

    def test_resolver_and_service_configuration_commands_are_escalated(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        for instruction in (
            "Run `resolvectl dns eth0 1.1.1.1`.",
            "Run `systemctl restart systemd-resolved`.",
        ):
            with self.subTest(instruction=instruction):
                result = _assess_recommendation_safety(instruction, transaction)
                self.assertEqual(result["decision"], "escalate")

    def test_wrapped_commands_are_detected_and_escalated_as_opaque(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        for instruction in (
            "Run `sh -c 'sudo apt remove package'`.",
            "Run `env sudo apt install package`.",
            "Run `python -c \"import os; os.remove('file')\"`.",
        ):
            with self.subTest(instruction=instruction):
                self.assertTrue(_contains_executable_command(instruction))
                result = _assess_recommendation_safety(instruction, transaction)
                self.assertEqual(result["decision"], "escalate")

    def test_safe_resolver_and_service_diagnostics_remain_allowed(self):
        transaction = _assess_apt_transaction("No transaction plan is available")
        for instruction in (
            "Run `resolvectl query archive.ubuntu.com`.",
            "Run `systemctl status systemd-resolved`.",
        ):
            with self.subTest(instruction=instruction):
                result = _assess_recommendation_safety(instruction, transaction)
                self.assertEqual(result["decision"], "allow")

    def test_model_cannot_hide_wrapped_mutation_in_answer_summary(self):
        draft = {
            "status": "recommendation",
            "summary": "Run `sh -c 'sudo apt install package'` to fix it.",
            "steps": [], "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)
        prepared = agent.prepare("Package installation issue", "Ubuntu 24.04")

        result = agent.generate(prepared)

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["safety_assessment"]["decision"], "escalate")
        self.assertNotIn("apt --fix-broken install", result["summary"])
        self.assertIn("removed from the summary", " ".join(result["review_reasons"]))

    def test_model_cannot_hide_risky_action_in_review_reason(self):
        draft = {
            "status": "needs_review",
            "summary": "Review the package issue.",
            "steps": [],
            "missing_information": [],
            "review_reasons": ["Run `sudo apt --fix-broken install` to repair it."],
        }
        agent, _ = _agent(draft)
        prepared = agent.prepare(
            "How can I inspect a package installation?",
            "I need general information about Ubuntu package management.",
        )

        result = agent.generate(prepared)

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["safety_assessment"]["decision"], "escalate")
        self.assertNotIn("apt --fix-broken install", " ".join(result["review_reasons"]))
        self.assertIn("effects are not established as safe", " ".join(result["review_reasons"]))

    def test_model_cannot_override_kernel_transaction_escalation(self):
        draft = {
            "status": "recommendation", "summary": "Repair the package state.",
            "steps": [{
                "instruction": "Review the package repair operation.",
                "command": "sudo apt --fix-broken install",
                "citations": [{"source_id": "OD-1", "quote": "package handling"}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        agent, chain = _agent(draft)
        prepared = agent.prepare(
            "Broken package dependencies on Ubuntu 22.04",
            "The following packages will be REMOVED:\n"
            "  linux-image-6.5.0-14-generic\n"
            "0 updated, 2 newly installed, 1 to be removed and 26 not updated.\n"
            "dpkg: linux-image-6.5.0-14-generic package processing error (--remove)\n"
            "E: Sub-process /usr/bin/dpkg returned an error code (1)",
        )

        result = agent.generate(prepared)

        chain.invoke.assert_called_once()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["steps"], [])
        self.assertTrue(any("kernel" in reason.lower() for reason in result["review_reasons"]))

    def test_model_cannot_hide_risky_action_in_summary(self):
        draft = {
            "status": "recommendation",
            "summary": "Run `sudo apt --fix-broken install` to repair the ticket.",
            "steps": [], "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft)
        prepared = agent.prepare(
            "Broken package dependencies on Ubuntu 22.04",
            "A package transaction result is not available.\n"
            "E: Sub-process /usr/bin/dpkg returned an error code (1)",
        )

        result = agent.generate(prepared)

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["safety_assessment"]["decision"], "escalate")

    def test_apt_explanation_in_summary_is_not_mistaken_for_a_package_action(self):
        draft = {
            "status": "needs_review",
            "summary": (
                "APT packages install to predefined system directories rather than "
                "letting the user choose an installation path."
            ),
            "steps": [],
            "missing_information": [],
            "review_reasons": [],
        }
        agent, _ = _agent(draft)

        result = agent.run(
            "Where does apt install applications?",
            "I want to understand package installation locations.",
        )

        summary_assessment = next(
            item for item in result["safety_assessment"]["recommendations"]
            if item["target"] == "summary"
        )
        self.assertEqual(summary_assessment["decision"], "allow")
        self.assertEqual(result["safety_assessment"]["decision"], "allow")

    def test_dangerous_command_in_retrieved_source_does_not_block_safe_diagnostic(self):
        source_text = (
            "Historical answer suggests `sudo rm /var/lib/dpkg/info/zoom.postrm`. "
            "Read-only audit command: `dpkg --audit`."
        )
        community = _CommunityRetriever()
        community.retrieve = lambda *_args, **_kwargs: {
            "evidence": [{
                "question_id": "777", "title": "Historical package repair",
                "answer_text": source_text, "rank": 1, "score": 0.9,
                "sources": {"answer": {"source_url": "https://askubuntu.com/a/778",
                                         "creation_date": "2024-01-01T00:00:00Z"}},
            }],
            "weak_evidence": False, "method": "dense",
        }
        draft = {
            "status": "recommendation", "summary": "Inspect package state.",
            "steps": [{
                "instruction": "Inspect package state using a dpkg audit and share the output.",
                "command": "dpkg --audit",
                "citations": [{"source_id": "AU-777", "quote": "dpkg --audit"}],
            }],
            "missing_information": [], "review_reasons": [],
        }
        agent, _ = _agent(draft, community=community)
        prepared = agent.prepare(
            "Check package state on Ubuntu 22.04",
            "Ubuntu 22.04 package issue",
        )

        result = agent.generate(prepared)

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(len(result["steps"]), 1)

    def test_generation_connection_failure_fails_closed_without_draft(self):
        agent, chain = _agent()
        chain.invoke.side_effect = ConnectionError("provider unavailable")
        prepared = agent.prepare(
            "Check DNS resolution on Ubuntu 22.04",
            "I need a read-only diagnostic for a hostname lookup.",
        )

        result = agent.generate(prepared)

        chain.invoke.assert_called_once()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["answer"], "")
        self.assertTrue(result["error"].startswith("ConnectionError: provider unavailable"))
        self.assertTrue(result["engineer_review_required"])

    def test_repeated_dns_configuration_step_is_withheld_when_ticket_says_it_was_tried(self):
        passage = (
            "To configure the resolver, add the IP addresses of the appropriate "
            "nameservers for your network to the netplan configuration file. "
            "The reviewed command is `sudo netplan apply`."
        )
        draft = {
            "status": "recommendation",
            "summary": "Review the DNS setup.",
            "steps": [{
                "instruction": "Configure the resolver with the appropriate nameservers.",
                "command": "sudo netplan apply",
                "citations": [{"source_id": "OD-1", "quote": passage}],
            }],
            "missing_information": [],
            "review_reasons": [],
        }
        official = _OfficialRetriever([_official_hit(text=passage)])
        agent, _ = _agent(draft, official=official)
        prepared = agent.prepare(
            "Ubuntu Server 22.04 DNS timeouts",
            "I installed Ubuntu 22.04 and I setup the interface with an IP and DNS servers. "
            "So I went ahead and added the DNS servers globally by adding them to "
            "/etc/systemd/resolved.conf and restarting the service. It still times out.",
            product_type="ubuntu_server", ubuntu_release="22.04",
        )

        result = agent.generate(prepared)

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["steps"], [])
        self.assertTrue(any(
            "already been attempted" in issue
            for issue in result["withheld_steps"][0]["issues"]
        ))

    def test_read_only_dns_diagnostic_remains_available_after_configuration_attempts(self):
        passage = "resolvectl status displays the current resolver state."
        draft = {
            "status": "recommendation",
            "summary": "Inspect current resolver state.",
            "steps": [{
                "instruction": "Inspect the current resolver state and share the output.",
                "command": "resolvectl status",
                "citations": [{"source_id": "OD-1", "quote": passage}],
            }],
            "missing_information": [],
            "review_reasons": [],
        }
        official = _OfficialRetriever([_official_hit(text=passage)])
        agent, _ = _agent(draft, official=official)
        prepared = agent.prepare(
            "Ubuntu Server 22.04 DNS timeouts",
            "I configured DNS servers in netplan and restarted systemd-resolved, but lookups still time out.",
            product_type="ubuntu_server", ubuntu_release="22.04",
        )

        result = agent.generate(prepared)

        self.assertEqual(result["step_validation"]["status"], "passed")
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["safety_assessment"]["decision"], "allow")

if __name__ == "__main__":
    unittest.main()
