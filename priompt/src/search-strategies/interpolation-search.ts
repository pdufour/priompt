import { SearchResult } from './index';
import { PriomptTokenizer } from '../tokenizer';
import { PromptElement } from '../types.d';
import { TooManyTokensForBasePriority, countTokensApproxFast_UNSAFE, countTokensExact, renderWithLevelAndEarlyExitWithTokenEstimation } from '../lib';

// Import shouldPrintVerboseLogs from lib.ts
const shouldPrintVerboseLogs = () => process.env.NODE_ENV === 'development' && process.env.PRINT_PRIOMPT_LOGS === "true";

/**
 * Runs an interpolation search to find the optimal priority level
 * This implementation has been improved to handle edge cases better and provide more robust search behavior
 */
export async function runInterpolationSearch(
	elem: PromptElement,
	sortedPriorityLevels: number[],
	tokenizer: PriomptTokenizer,
	tokenLimit: number,
	usedTokenlimit: number,
	countTokensFast_UNSAFE: boolean | undefined,
	lastMessageIsIncomplete: boolean | undefined
): Promise<SearchResult> {
	// Basic setup
	let largestTokenCountSeen = 0;
	let exclusiveLowerBound = -1;
	let inclusiveUpperBound = 0;
	const startTime = performance.now();

	// Early return for empty array
	if (sortedPriorityLevels.length === 0) {
		return { exclusiveLowerBound, inclusiveUpperBound, largestTokensSeen: largestTokenCountSeen };
	}

	if (shouldPrintVerboseLogs()) {
		console.debug(`[InterpolationSearch] Starting search with ${sortedPriorityLevels.length} priority levels`);
	}

	// Cache to store token counts for each tested level
	const tokenCounts = new Map<number, number>();
	// Keep track of tested indices to avoid redundant tests
	const testedIndices = new Set<number>();

	// Helper function to test priority level and cache results
	async function testLevel(index: number): Promise<boolean> {
		if (index < 0 || index >= sortedPriorityLevels.length) {
			return false;
		}

		testedIndices.add(index);
		const level = sortedPriorityLevels[index];

		if (tokenCounts.has(level)) {
			return tokenCounts.get(level)! <= usedTokenlimit;
		}

		try {
			const prompt = renderWithLevelAndEarlyExitWithTokenEstimation(elem, level, tokenizer, tokenLimit);

			if (!prompt || prompt.prompt === undefined) {
				tokenCounts.set(level, Number.MAX_SAFE_INTEGER);
				return false;
			}

			const tokens = countTokensFast_UNSAFE
				? await countTokensApproxFast_UNSAFE(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete })
				: await countTokensExact(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete });

			largestTokenCountSeen = Math.max(largestTokenCountSeen, tokens);
			const totalTokens = tokens + (prompt.emptyTokenCount || 0);
			tokenCounts.set(level, totalTokens);

			const isValid = totalTokens <= usedTokenlimit;

			if (shouldPrintVerboseLogs()) {
				console.debug(`[InterpolationSearch] Tested level ${level}: ${totalTokens} tokens (${isValid ? 'valid' : 'invalid'})`);
			}

			return isValid;
		} catch (e) {
			tokenCounts.set(level, Number.MAX_SAFE_INTEGER);
			return false;
		}
	}

	// Pure interpolation search algorithm, simplified
	let low = 0;
	let high = sortedPriorityLevels.length - 1;

	// First check if the highest priority level fits
	// This is typically the one with the lowest token count
	if (!await testLevel(high)) {
		// If even the highest priority doesn't fit, then nothing will
		return { exclusiveLowerBound: -1, inclusiveUpperBound: 0, largestTokensSeen: largestTokenCountSeen };
	}

	// Next check if the lowest priority level fits
	if (await testLevel(low)) {
		// If the lowest priority fits, it's the best one to use
		return { exclusiveLowerBound: -1, inclusiveUpperBound: low, largestTokensSeen: largestTokenCountSeen };
	}

	// At this point, we know:
	// - high priority level (at index high) fits
	// - low priority level (at index low) doesn't fit
	// We need to find the boundary between them

	// Start with the highest valid priority level we know
	let bestFittingIndex = high;

	// Limit the number of iterations to prevent excessive testing
	const MAX_ITERATIONS = 10;
	let iterations = 0;

	while (low < high - 1 && iterations < MAX_ITERATIONS) {
		iterations++;

		// Get token counts for interpolation
		const lowTokens = tokenCounts.get(sortedPriorityLevels[low])!;
		const highTokens = tokenCounts.get(sortedPriorityLevels[high])!;

		// Calculate the next position to test using interpolation
		let mid: number;

		// Only use real interpolation if we have a valid token relationship
		if (lowTokens > highTokens && lowTokens > usedTokenlimit) {
			// Calculate position based on where the token limit likely falls
			const ratio = (usedTokenlimit - highTokens) / (lowTokens - highTokens);
			// Ensure the ratio is within reasonable bounds
			const boundedRatio = Math.max(0.1, Math.min(0.9, ratio));
			mid = Math.floor(high - boundedRatio * (high - low));
		} else {
			// Fall back to binary search
			mid = Math.floor((low + high) / 2);
		}

		// Skip if already tested
		if (testedIndices.has(mid)) {
			// Simple approach - just move to next index
			let newMid = mid;
			while (testedIndices.has(newMid) && newMid < high - 1) {
				newMid++;
			}

			if (newMid === mid || testedIndices.has(newMid)) {
				// Couldn't find an untested index going forward, try backward
				newMid = mid;
				while (testedIndices.has(newMid) && newMid > low + 1) {
					newMid--;
				}

				if (newMid === mid || testedIndices.has(newMid)) {
					// All indices in range already tested
					break;
				}
			}

			mid = newMid;
		}

		// Test the interpolated position
		const midFits = await testLevel(mid);

		if (midFits) {
			// If this level fits, it's a candidate for the result
			// We also move our top bound down
			high = mid;
			bestFittingIndex = mid;
		} else {
			// If it doesn't fit, move the bottom bound up
			low = mid;
		}
	}

	// The best fitting index is the one we've found
	inclusiveUpperBound = bestFittingIndex;

	// Determine the exclusive lower bound (first level that doesn't fit below the best fitting one)
	if (bestFittingIndex > 0) {
		for (let i = bestFittingIndex - 1; i >= 0; i--) {
			if (testedIndices.has(i)) {
				// Use cached result
				if (tokenCounts.get(sortedPriorityLevels[i])! > usedTokenlimit) {
					exclusiveLowerBound = i;
					break;
				}
			} else {
				// Test this level
				if (!(await testLevel(i))) {
					exclusiveLowerBound = i;
					break;
				}
			}
		}
	}

	if (shouldPrintVerboseLogs()) {
		const endTime = performance.now();
		console.debug(`[InterpolationSearch] Completed in ${(endTime - startTime).toFixed(1)}ms (${iterations} iterations)`);
		console.debug(`[InterpolationSearch] Result: level ${sortedPriorityLevels[inclusiveUpperBound]} at index ${inclusiveUpperBound}`);
	}

	return { exclusiveLowerBound, inclusiveUpperBound, largestTokensSeen: largestTokenCountSeen };
}