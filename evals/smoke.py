"""
Smoke test of the research workflow against the real APIs.

It runs a few simple questions with minimal research budgets, so that a run fits the free tier of the Gemini API,
and checks properties that need no LLM judge: the run completes, every step produces findings, every citation
points to a listed reference. It also records model calls, tokens and time.

Reports are saved as `<out>/<system>/idx-<n>.md`, the layout DeepResearch Bench II expects, so the same runner
can feed a full evaluation later.

Usage:
    uv run python evals/smoke.py [--system NAME] [--questions FILE] [--limit N] [--full-budget]
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / 'multi_tool_agent' / '.env')

# Minimal budgets: one round of at most two researchers with 4 search/read calls each (about 8-15 model calls per
# question). The number of search candidates is not reduced, since it costs no model calls
MINIMAL_BUDGET = {
    'MAX_STEPS': '2',
    'MAX_ITERATIONS': '1',
    'RESEARCHER_TOOL_CALLS': '4',
    'WEB_RESULTS': '3',
}

DEFAULT_QUESTIONS = [
    # Current fact: one web step
    'Who won the most recent Nobel Prize in Physics, and for what discovery?',
    # A well-known paper: arXiv
    'What is the main idea of the LoRA paper (Low-Rank Adaptation of Large Language Models)?',
    # Mixed: papers and current coverage
    'How does speculative decoding speed up LLM inference, and which serving frameworks support it today?',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--system', default='deep-research',
        help='Name of the system, used as output folder',
    )
    parser.add_argument(
        '--questions', type=Path,
        help='Text file with one question per line',
    )
    parser.add_argument(
        '--limit', type=int,
        help='Run only the first N questions',
    )
    parser.add_argument(
        '--out', type=Path, default=ROOT /
        'evals' / 'out', help='Output folder',
    )
    parser.add_argument(
        '--full-budget', action='store_true',
        help='Use the configured budgets, not the minimal ones',
    )
    return parser.parse_args()


def citations(text: str) -> set[int]:
    return {
        int(number)
        for group in re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', text)
        for number in group.split(',')
    }


async def run_question(question: str) -> dict[str, Any]:
    """Run one question in a fresh session and collect the answer, the references, the steps and the usage."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from multi_tool_agent.agent import root_agent
    from multi_tool_agent.core.models import StepResult

    runner = InMemoryRunner(agent=root_agent, app_name='smoke')
    session = await runner.session_service.create_session(app_name='smoke', user_id='smoke')
    usage = {'model_calls': 0, 'prompt_tokens': 0, 'output_tokens': 0}
    last_root_message = ''
    started = time.monotonic()
    error = None
    try:
        content = types.Content(role='user', parts=[types.Part(text=question)])
        async for event in runner.run_async(user_id='smoke', session_id=session.id, new_message=content):
            if event.usage_metadata:
                usage['model_calls'] += 1
                usage['prompt_tokens'] += event.usage_metadata.prompt_token_count or 0
                usage['output_tokens'] += event.usage_metadata.candidates_token_count or 0
            if event.author == 'root' and event.content and event.content.parts:
                last_root_message = ''.join(
                    part.text or '' for part in event.content.parts
                )
    except Exception as e:
        error = f'{type(e).__name__}: {e}'

    session = await runner.session_service.get_session(app_name='smoke', user_id='smoke', session_id=session.id)
    last_run = session.state.get('last_run')
    return {
        'question': question,
        'researched': last_run is not None,
        'answer': (last_run or {}).get('answer', ''),
        'references': last_root_message if last_root_message.startswith('**References**') else '',
        'steps': [StepResult.model_validate(step) for step in (last_run or {}).get('steps', [])],
        'error': error,
        'seconds': round(time.monotonic() - started),
        **usage,
    }


def warnings(result: dict[str, Any]) -> list[str]:
    """Return the non-blocking issues of a run: tasks that found nothing while others did."""
    return [f'task `{step.step.objective}` found nothing' for step in result['steps'] if not step.findings]


def check(result: dict[str, Any]) -> list[str]:
    """Return the failed checks of a run."""
    failures = []
    if result['error']:
        failures.append(f'run failed: {result["error"]}')
    if not result['researched']:
        failures.append(
            'no research was run (the question was not classified as research)',
        )
    if not result['answer']:
        failures.append('empty answer')
    if not result['steps']:
        failures.append('no research steps')
    for step in result['steps']:
        if step.error:
            failures.append(
                f'task `{step.step.objective}` failed: {step.error}',
            )
    if result['steps'] and not any(step.findings for step in result['steps']):
        failures.append('no researcher found anything')
    cited = citations(result['answer'])
    listed = citations(
        re.sub(r'^\*\*References\*\*', '', result['references']),
    )
    if not cited:
        failures.append('no citations in the answer')
    if cited - listed:
        failures.append(
            f'citations without a reference: {sorted(cited - listed)}',
        )
    return failures


async def main() -> int:
    args = parse_args()
    if not args.full_budget:
        os.environ.update(MINIMAL_BUDGET)
    questions = (
        [
            line.strip()
            for line in args.questions.read_text().splitlines() if line.strip()
        ]
        if args.questions else DEFAULT_QUESTIONS
    )[:args.limit]

    out_dir = args.out / args.system
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for idx, question in enumerate(questions, start=1):
        print(f'[{idx}/{len(questions)}] {question}', flush=True)
        result = await run_question(question)
        failures = check(result)
        (out_dir / f'idx-{idx}.md').write_text(
            f'{result["answer"]}\n\n{result["references"]}\n',
        )
        summary.append({
            'idx': idx,
            'question': question,
            'passed': not failures,
            'failures': failures,
            'warnings': warnings(result),
            'steps': [
                {
                    'objective': step.step.objective, 'source_types': step.step.sources, 'lens': step.step.lens,
                    'notes': step.notes,
                    'sources': len(step.findings), 'findings': sum(map(len, step.findings.values())),
                    'brief': step.brief,
                    'error': step.error,
                }
                for step in result['steps']
            ],
            **{key: result[key] for key in ('seconds', 'model_calls', 'prompt_tokens', 'output_tokens')},
        })
        status = 'PASS' if not failures else 'FAIL: ' + '; '.join(failures)
        print(
            f'    {status} | {result["seconds"]}s, {result["model_calls"]} model calls, '
            f'{result["prompt_tokens"]}+{result["output_tokens"]} tokens',
            flush=True,
        )

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    passed = sum(item['passed'] for item in summary)
    print(f'\n{passed}/{len(summary)} passed. Reports and summary.json in {out_dir}')
    return 0 if passed == len(summary) else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
