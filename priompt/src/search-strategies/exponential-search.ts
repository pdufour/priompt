import { SearchResult } from './index';
import { PriomptTokenizer } from '../tokenizer';
import { PromptElement } from '../types.d';
import { countTokensApproxFast_UNSAFE, countTokensExact, renderWithLevelAndEarlyExitWithTokenEstimation } from '../lib';
import { runBinarySearch } from './binary-search';

// Import shouldPrintVerboseLogs from lib.ts
const shouldPrintVerboseLogs = () => process.env.NODE_ENV === 'development' && process.env.PRINT_PRIOMPT_LOGS === "true";

/**
 * Runs an exponential search followed by a binary search in a narrower range
 */
export async function runExponentialSearch(
	elem: PromptElement,
	sortedPriorityLevels: number[],
	tokenizer: PriomptTokenizer,
	tokenLimit: number,
	usedTokenlimit: number,
	countTokensFast_UNSAFE: boolean | undefined,
	lastMessageIsIncomplete: boolean | undefined
): Promise<SearchResult> {
	let largestTokenCountSeen = 0;
	let exclusiveLowerBound = -1;
	let inclusiveUpperBound = sortedPriorityLevels.length - 1;

	// EXPONENTIAL BINARY SEARCH
	// First phase: exponential stride to find upper bound
	let stride = 1;
	let index = 0;
	let upperBound = -1; // Start with -1 to indicate "not found yet"
	let foundFittingLevel = false;

	while (index < sortedPriorityLevels.length) {
		const priorityLevel = sortedPriorityLevels[index];
		let start: number | undefined;

		if (shouldPrintVerboseLogs()) {
			console.debug(`Exponential search - Trying priority level ${priorityLevel} with index ${index}`)
			start = performance.now();
		}

		let tokenCount = -1;
		let countStart: number | undefined;

		try {
			const prompt = renderWithLevelAndEarlyExitWithTokenEstimation(elem, priorityLevel, tokenizer, tokenLimit);
			countStart = performance.now();

			if (countTokensFast_UNSAFE === true) {
				tokenCount = await countTokensApproxFast_UNSAFE(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete });
			} else {
				tokenCount = await countTokensExact(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete });
			}

			largestTokenCountSeen = Math.max(largestTokenCountSeen, tokenCount);

			if (tokenCount + prompt.emptyTokenCount <= usedTokenlimit) {
				// This fits, remember this index as our upperBound
				upperBound = index;
				foundFittingLevel = true;

				// Exponentially increase stride to find next test point
				index += stride;
				stride *= 2;
			} else {
				// Found a level that doesn't fit, break to binary search
				break;
			}
		} catch {
			// Level is too low, move to next with exponential stride
			index += stride;
			stride *= 2;
		} finally {
			if (shouldPrintVerboseLogs()) {
				const end = performance.now();
				console.debug(`Exponential search - Priority ${priorityLevel} with index ${index - stride / 2} took ${end - (start ?? 0)} ms and has ${tokenCount} tokens (counting took ${end - (countStart ?? 0)})`);
			}
		}
	}

	// If we didn't find a fitting level with exponential search, fall back to standard binary search
	if (!foundFittingLevel) {
		if (shouldPrintVerboseLogs()) {
			console.debug(`No fitting level found with exponential search, falling back to standard binary search`);
		}
		// Reset bounds to the default for binary search
		return await runBinarySearch(elem, sortedPriorityLevels, tokenizer, tokenLimit, usedTokenlimit, countTokensFast_UNSAFE, lastMessageIsIncomplete);
	} else {
		// Second phase: binary search in the range [upperBound, min(upperBound+stride/2, length)]
		const lowerBound = upperBound;
		const newUpperBound = Math.min(upperBound + stride / 2, sortedPriorityLevels.length - 1);

		// Set the bounds for the binary search
		exclusiveLowerBound = lowerBound - 1;  // Make it exclusive
		inclusiveUpperBound = newUpperBound;

		if (shouldPrintVerboseLogs()) {
			console.debug(`Switching to binary search in range [${exclusiveLowerBound + 1}, ${inclusiveUpperBound}]`);
		}

		// Run binary search in the narrowed range
		// Create a view of the sortedPriorityLevels array for the narrowed range
		const narrowedRange = sortedPriorityLevels.slice(lowerBound, newUpperBound + 1);
		const offsetFromOriginal = lowerBound;

		// If the narrowed range is too small, just use the results we have
		if (narrowedRange.length <= 2) {
			return {
				exclusiveLowerBound,
				inclusiveUpperBound: lowerBound, // Use the known good level
				largestTokensSeen: largestTokenCountSeen
			};
		}

		// Run binary search on the narrowed range
		const result = await runBinarySearch(elem, narrowedRange, tokenizer, tokenLimit, usedTokenlimit, countTokensFast_UNSAFE, lastMessageIsIncomplete);

		// Translate the result back to the original array indices
		return {
			exclusiveLowerBound: result.exclusiveLowerBound < 0 ? exclusiveLowerBound : result.exclusiveLowerBound + offsetFromOriginal,
			inclusiveUpperBound: result.inclusiveUpperBound + offsetFromOriginal,
			largestTokensSeen: Math.max(largestTokenCountSeen, result.largestTokensSeen)
		};
	}
}