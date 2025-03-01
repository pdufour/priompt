import { PriomptTokenizer } from '../tokenizer';
import { PromptElement } from '../types.d';
import { SearchResult } from './index';
import { runBinarySearch } from './binary-search';
import { runExponentialSearch } from './exponential-search';
import { runInterpolationSearch } from './interpolation-search';
import { runGeneticSearch } from './genetic-search';
import { shouldPrintVerboseLogs } from '../lib';

/**
 * Selects and runs the appropriate search strategy based on the provided strategy name
 */
export async function runSearchStrategy(
	strategy: string | undefined,
	elem: PromptElement,
	sortedPriorityLevels: number[],
	tokenizer: PriomptTokenizer,
	tokenLimit: number,
	usedTokenlimit: number,
	countTokensFast_UNSAFE: boolean | undefined,
	lastMessageIsIncomplete: boolean | undefined
): Promise<SearchResult> {
	if (shouldPrintVerboseLogs()) {
		console.debug(`Running search strategy: "${strategy || 'default (exponential-then-binary-search)'}" with ${sortedPriorityLevels.length} priority levels`);
	}

	if (strategy === "binary") {
		if (shouldPrintVerboseLogs()) {
			console.debug(`Using Binary search strategy`);
		}
		return runBinarySearch(
			elem,
			sortedPriorityLevels,
			tokenizer,
			tokenLimit,
			usedTokenlimit,
			countTokensFast_UNSAFE,
			lastMessageIsIncomplete
		);
	} else if (strategy === "exponential-then-binary-search") {
		if (shouldPrintVerboseLogs()) {
			console.debug(`Using Exponential-then-Binary search strategy${strategy === "exponential-then-binary-search" ? " (legacy name)" : ""}`);
		}
		return runExponentialSearch(
			elem,
			sortedPriorityLevels,
			tokenizer,
			tokenLimit,
			usedTokenlimit,
			countTokensFast_UNSAFE,
			lastMessageIsIncomplete
		);
	} else if (strategy === "interpolation") {
		if (shouldPrintVerboseLogs()) {
			console.debug(`Using Interpolation search strategy`);
		}
		return runInterpolationSearch(
			elem,
			sortedPriorityLevels,
			tokenizer,
			tokenLimit,
			usedTokenlimit,
			countTokensFast_UNSAFE,
			lastMessageIsIncomplete
		);
	} else if (strategy === "genetic-algorithim") {
		if (shouldPrintVerboseLogs()) {
			console.debug(`Using Genetic Algorithm search strategy`);
		}
		return runGeneticSearch(
			elem,
			sortedPriorityLevels,
			tokenizer,
			tokenLimit,
			usedTokenlimit,
			countTokensFast_UNSAFE,
			lastMessageIsIncomplete
		);
	}

	// Default to exponential-then-binary search if no valid strategy is specified
	if (shouldPrintVerboseLogs()) {
		console.debug(`No valid strategy specified, defaulting to Exponential-then-Binary search`);
	}
	return runExponentialSearch(
		elem,
		sortedPriorityLevels,
		tokenizer,
		tokenLimit,
		usedTokenlimit,
		countTokensFast_UNSAFE,
		lastMessageIsIncomplete
	);
}