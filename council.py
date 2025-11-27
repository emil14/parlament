"""3-stage LLM Council orchestration."""

import re
from collections import defaultdict
from dataclasses import dataclass

from llm_client import query_models_parallel, query_model
from config import Settings


@dataclass
class Stage2Result:
    """Result from Stage 2 ranking."""
    model: str
    ranking: str
    parsed_ranking: list[str]


@dataclass
class Stage3Result:
    """Final synthesized result from Stage 3."""
    model: str
    response: str


@dataclass
class AggregateRanking:
    """Aggregate ranking for a model across all rankings."""
    model: str
    average_rank: float
    rankings_count: int


@dataclass
class Stage2Rankings:
    """Complete Stage 2 results including rankings and label mapping."""
    results: list[Stage2Result]
    label_to_model: dict[str, str]


@dataclass
class CouncilMetadata:
    """Metadata from the council process."""
    label_to_model: dict[str, str]
    aggregate_rankings: list[AggregateRanking]


@dataclass
class CouncilResult:
    """Complete result from running the full council process."""
    stage1_responses: dict[str, str]
    stage2_rankings: list[Stage2Result]
    stage3_result: Stage3Result
    metadata: CouncilMetadata
    total_usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens


async def stage1_collect_responses(
    user_query: str,
    settings: Settings
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question
        settings: Application settings

    Returns:
        Tuple of (dict mapping model name to response content, usage dict)
    """
    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    try:
        responses = await query_models_parallel(settings.COUNCIL_MODELS, messages, settings)
    except RuntimeError as e:
        raise RuntimeError(f"Stage 1 (collecting responses) failed: {str(e)}") from e

    # Extract response content and accumulate usage
    stage1_responses = {
        model: response.get('content', '')
        for model, response in responses.items()
    }
    
    # Accumulate usage from all models in Stage 1
    stage1_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    for _model, response in responses.items():
        usage = response.get('usage', {})
        stage1_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
        stage1_usage['completion_tokens'] += usage.get('completion_tokens', 0)
        stage1_usage['total_tokens'] += usage.get('total_tokens', 0)
    
    return stage1_responses, stage1_usage


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: dict[str, str],
    settings: Settings
) -> tuple[Stage2Rankings, dict[str, int]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Dict mapping model name to response content from Stage 1
        settings: Application settings

    Returns:
        Stage2Rankings containing results and label mapping
    """
    # Create anonymized labels for responses (Response A, Response B, etc.)
    models = list(stage1_results.keys())
    labels = [chr(65 + i) for i in range(len(models))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model: dict[str, str] = {
        f"Response {label}": model
        for label, model in zip(labels, models)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{stage1_results[model]}"
        for label, model in zip(labels, models)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:

1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.

2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:

- Start with the line "FINAL RANKING:" (all caps, with colon)

- Then list the responses from best to worst as a numbered list

- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")

- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...

Response B is accurate but lacks depth on Z...

Response C offers the most comprehensive answer...

FINAL RANKING:

1. Response C

2. Response A

3. Response B

Now provide your evaluation and ranking:"""

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    try:
        responses = await query_models_parallel(settings.COUNCIL_MODELS, messages, settings)
    except RuntimeError as e:
        raise RuntimeError(f"Stage 2 (collecting rankings) failed: {str(e)}") from e

    # Format results and accumulate usage
    stage2_results: list[Stage2Result] = []
    stage2_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    for model, response in responses.items():
        full_text = response.get('content', '')
        parsed = parse_ranking_from_text(full_text)
        stage2_results.append(Stage2Result(
            model=model,
            ranking=full_text,
            parsed_ranking=parsed
        ))
        
        # Accumulate usage
        usage = response.get('usage', {})
        stage2_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
        stage2_usage['completion_tokens'] += usage.get('completion_tokens', 0)
        stage2_usage['total_tokens'] += usage.get('total_tokens', 0)

    return (
        Stage2Rankings(
            results=stage2_results,
            label_to_model=label_to_model
        ),
        stage2_usage
    )


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: dict[str, str],
    stage2_results: list[Stage2Result],
    settings: Settings
) -> tuple[Stage3Result, dict[str, int]]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Dict mapping model name to response content from Stage 1
        stage2_results: Rankings from Stage 2
        settings: Application settings

    Returns:
        Stage3Result with model and response
    """
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {model}\nResponse: {response}"
        for model, response in stage1_results.items()
    ])

    stage2_text = "\n\n".join([
        f"Model: {result.model}\nRanking: {result.ranking}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:

{stage1_text}

STAGE 2 - Peer Rankings:

{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:

- The individual responses and their insights

- The peer rankings and what they reveal about response quality

- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    try:
        response = await query_model(settings.CHAIRMAN_MODEL, messages, settings)
    except RuntimeError as e:
        raise RuntimeError(f"Stage 3 (Chairman synthesis) failed for model '{settings.CHAIRMAN_MODEL}': {str(e)}") from e

    # Extract usage from Stage 3
    usage = response.get('usage', {})
    stage3_usage = {
        'prompt_tokens': usage.get('prompt_tokens', 0),
        'completion_tokens': usage.get('completion_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
    }

    return (
        Stage3Result(
            model=settings.CHAIRMAN_MODEL,
            response=response.get('content', '')
        ),
        stage3_usage
    )


def parse_ranking_from_text(ranking_text: str) -> list[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of response labels in ranked order
    """
    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                result: list[str] = []
                for m in numbered_matches:
                    match = re.search(r'Response [A-Z]', m)
                    if match:
                        result.append(match.group())
                if result:
                    return result

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: list[Stage2Result],
    label_to_model: dict[str, str]
) -> list[AggregateRanking]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of AggregateRanking sorted best to worst (by average rank)
    """
    # Track positions for each model
    model_positions: defaultdict[str, list[int]] = defaultdict(list)

    for result in stage2_results:
        # Use the already-parsed ranking from Stage 2
        parsed_ranking = result.parsed_ranking

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate: list[AggregateRanking] = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append(AggregateRanking(
                model=model,
                average_rank=round(avg_rank, 2),
                rankings_count=len(positions)
            ))

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x.average_rank)

    return aggregate


async def generate_conversation_title(
    user_query: str,
    settings: Settings
) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message
        settings: Application settings

    Returns:
        A short title (3-5 words)
    """
    title_prompt = f"""Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

    messages = [{"role": "user", "content": title_prompt}]

    # Use gemini-2.5-flash for title generation (fast and cheap)
    try:
        response = await query_model("google/gemini-2.5-flash", messages, settings, timeout=30.0)
    except RuntimeError as e:
        raise RuntimeError(f"Title generation failed for model 'google/gemini-2.5-flash': {str(e)}") from e

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(
    user_query: str,
    settings: Settings
) -> CouncilResult:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question
        settings: Application settings

    Returns:
        CouncilResult containing all stage results and metadata
    """
    # Stage 1: Collect individual responses
    stage1_responses, stage1_usage = await stage1_collect_responses(user_query, settings)

    # Stage 2: Collect rankings
    stage2_rankings, stage2_usage = await stage2_collect_rankings(
        user_query,
        stage1_responses,
        settings
    )

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(
        stage2_rankings.results,
        stage2_rankings.label_to_model
    )

    # Stage 3: Synthesize final answer
    stage3_result, stage3_usage = await stage3_synthesize_final(
        user_query,
        stage1_responses,
        stage2_rankings.results,
        settings
    )

    # Accumulate total usage across all stages
    total_usage = {
        'prompt_tokens': stage1_usage['prompt_tokens'] + stage2_usage['prompt_tokens'] + stage3_usage['prompt_tokens'],
        'completion_tokens': stage1_usage['completion_tokens'] + stage2_usage['completion_tokens'] + stage3_usage['completion_tokens'],
        'total_tokens': stage1_usage['total_tokens'] + stage2_usage['total_tokens'] + stage3_usage['total_tokens'],
    }

    # Prepare metadata
    metadata = CouncilMetadata(
        label_to_model=stage2_rankings.label_to_model,
        aggregate_rankings=aggregate_rankings
    )

    return CouncilResult(
        stage1_responses=stage1_responses,
        stage2_rankings=stage2_rankings.results,
        stage3_result=stage3_result,
        metadata=metadata,
        total_usage=total_usage
    )

