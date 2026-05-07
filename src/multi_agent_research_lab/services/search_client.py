"""Search client abstraction for ResearcherAgent."""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Search client with Tavily support and built-in mock fallback.

    Uses Tavily API when TAVILY_API_KEY is set, otherwise falls back to
    a mock search that returns plausible placeholder results so the full
    pipeline can run without external dependencies.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._tavily_key = settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        if self._tavily_key:
            return self._tavily_search(query, max_results)
        logger.warning("No TAVILY_API_KEY set — using mock search results.")
        return self._mock_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Real search via Tavily API."""
        from tavily import TavilyClient  # lazy import

        client = TavilyClient(api_key=self._tavily_key)
        response = client.search(query=query, max_results=max_results)

        results: list[SourceDocument] = []
        for item in response.get("results", []):
            results.append(SourceDocument(
                title=item.get("title", "Untitled"),
                url=item.get("url"),
                snippet=item.get("content", ""),
                metadata={"score": item.get("score", 0.0)},
            ))
        logger.info("Tavily search returned %d results for: %s", len(results), query)
        return results

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return plausible mock results for offline/demo usage."""

        mock_sources = [
            SourceDocument(
                title=f"Research Paper: Advances in {query[:50]}",
                url="https://arxiv.org/abs/example-001",
                snippet=f"This paper presents recent advances in {query}. "
                        "Key findings include improved performance through novel architectures "
                        "and training methodologies that achieve state-of-the-art results.",
                metadata={"source": "mock", "score": 0.95},
            ),
            SourceDocument(
                title=f"Survey: A Comprehensive Overview of {query[:40]}",
                url="https://arxiv.org/abs/example-002",
                snippet=f"A comprehensive survey covering the landscape of {query}. "
                        "The survey categorizes existing approaches into three main families "
                        "and identifies key open challenges for future research.",
                metadata={"source": "mock", "score": 0.90},
            ),
            SourceDocument(
                title=f"Industry Report: {query[:40]} in Production",
                url="https://example.com/industry-report",
                snippet=f"This report analyzes real-world deployments of {query}. "
                        "Production systems show 40% improvement in efficiency when "
                        "adopting multi-agent patterns with proper guardrails.",
                metadata={"source": "mock", "score": 0.85},
            ),
            SourceDocument(
                title=f"Tutorial: Building Systems with {query[:35]}",
                url="https://example.com/tutorial",
                snippet=f"A hands-on tutorial for implementing {query}. "
                        "Covers architecture design, failure handling, and evaluation "
                        "strategies for production deployment.",
                metadata={"source": "mock", "score": 0.80},
            ),
            SourceDocument(
                title=f"Blog: Lessons Learned from {query[:35]}",
                url="https://example.com/blog-post",
                snippet=f"Practical lessons learned from deploying {query} at scale. "
                        "Key takeaways include the importance of clear role separation, "
                        "shared state design, and comprehensive observability.",
                metadata={"source": "mock", "score": 0.75},
            ),
        ]
        results = mock_sources[:max_results]
        logger.info("Mock search returned %d results for: %s", len(results), query)
        return results
