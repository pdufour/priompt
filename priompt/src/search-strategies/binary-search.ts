import { SearchResult } from './index';
import { PriomptTokenizer } from '../tokenizer';
import { PromptElement } from '../types.d';
import { countTokensApproxFast_UNSAFE, countTokensExact, renderWithLevelAndEarlyExitWithTokenEstimation } from '../lib';

// Import shouldPrintVerboseLogs from lib.ts
const shouldPrintVerboseLogs = () => process.env.NODE_ENV === 'development' && process.env.PRINT_PRIOMPT_LOGS === "true";

/**
 * Runs a standard binary search to find the optimal priority level
 */
export async function runBinarySearch(
	elem: PromptElement,
	sortedPriorityLevels: number[],
	tokenizer: PriomptTokenizer,
	tokenLimit: number,
	usedTokenlimit: number,
	countTokensFast_UNSAFE: boolean | undefined,
	lastMessageIsIncomplete: boolean | undefined,
	initialLargestTokenCountSeen: number = 0,
	initialTokenCounts: Map<number, number> = new Map()
): Promise<SearchResult> {
	let largestTokenCountSeen = initialLargestTokenCountSeen;
	let exclusiveLowerBound = -1;
	let inclusiveUpperBound = sortedPriorityLevels.length - 1;

	// Cache to store token counts
	const tokenCounts = new Map<number, number>(initialTokenCounts);

	// Standard binary search phase
	let binarySearchFailed = false;
	while (exclusiveLowerBound < inclusiveUpperBound - 1) {
		const candidateLevelIndex = Math.floor((exclusiveLowerBound + inclusiveUpperBound) / 2);
		const candidateLevel = sortedPriorityLevels[candidateLevelIndex];
		let start: number | undefined;
		if (shouldPrintVerboseLogs()) {
			console.debug(`Binary search - Trying candidate level ${candidateLevel} with index ${candidateLevelIndex}`)
			start = performance.now();
		}
		let countStart: number | undefined;
		let tokenCount = -1;
		try {
			// Check cache first
			if (tokenCounts.has(candidateLevel)) {
				tokenCount = tokenCounts.get(candidateLevel)!;
				if (tokenCount + 0 > usedTokenlimit) { // 0 is a placeholder for emptyTokenCount
					// this means that the candidateLevel is too low
					exclusiveLowerBound = candidateLevelIndex;
				} else {
					// this means the candidate level is too high or it is just right
					inclusiveUpperBound = candidateLevelIndex;
				}
			} else {
				const prompt = renderWithLevelAndEarlyExitWithTokenEstimation(elem, candidateLevel, tokenizer, tokenLimit);
				countStart = performance.now();

				if (countTokensFast_UNSAFE === true) {
					tokenCount = await countTokensApproxFast_UNSAFE(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete });
				} else {
					tokenCount = await countTokensExact(tokenizer, prompt.prompt ?? "", { lastMessageIsIncomplete });
				}

				largestTokenCountSeen = Math.max(largestTokenCountSeen, tokenCount);
				const totalTokens = tokenCount + (prompt.emptyTokenCount || 0);
				tokenCounts.set(candidateLevel, totalTokens);

				if (totalTokens > usedTokenlimit) {
					// this means that the candidateLevel is too low
					exclusiveLowerBound = candidateLevelIndex;
				} else {
					// this means the candidate level is too high or it is just right
					inclusiveUpperBound = candidateLevelIndex;
				}
			}
		} catch {
			// this means the candidate level is too low
			exclusiveLowerBound = candidateLevelIndex;
		} finally {
			if (shouldPrintVerboseLogs()) {
				const end = performance.now();
				console.debug(`Binary search - Candidate level ${candidateLevel} with index ${candidateLevelIndex} took ${end - (start ?? 0)} ms and has ${tokenCount} tokens(-1 means early exit, counting took ${end - (countStart ?? 0)})`);
			}
		}
	}

	// Verify that we have a valid solution
	// If exclusiveLowerBound == sortedPriorityLevels.length - 1, then binary search failed to find a valid level
	if (exclusiveLowerBound === sortedPriorityLevels.length - 1) {
		binarySearchFailed = true;
		if (shouldPrintVerboseLogs()) {
			console.debug(`Binary search failed to find a priority level that fits within token limit`);
		}
		// Try the highest priority level (most restrictive) as a last resort
		inclusiveUpperBound = 0;
	}

	return {
		exclusiveLowerBound,
		inclusiveUpperBound,
		largestTokensSeen: largestTokenCountSeen
	};
}

/**
 * Creates bins from the sorted priority levels
 * @param sortedLevels The sorted array of priority levels
 * @param binSize The size of each bin
 * @returns An array of bins, where each bin is an array of priority levels
 */
function createBins(sortedLevels: number[], binSize: number): number[][] {
	const bins: number[][] = [];
	for (let i = 0; i < sortedLevels.length; i += binSize) {
		bins.push(sortedLevels.slice(i, i + binSize));
	}
	return bins;
}

/**
 * Runs a binned binary search to find the optimal priority level
 * First divides the priority levels into bins, then performs binary search
 * to find the right bin, and finally does a detailed search within that bin
 */
export async function runBinnedBinarySearch(
	elem: PromptElement,
	sortedPriorityLevels: number[],
	tokenizer: PriomptTokenizer,
	tokenLimit: number,
	usedTokenlimit: number,
	countTokensFast_UNSAFE: boolean | undefined,
	lastMessageIsIncomplete: boolean | undefined
): Promise<SearchResult> {
	let largestTokenCountSeen = 0;

	// Early exit for empty array
	if (sortedPriorityLevels.length === 0) {
		if (shouldPrintVerboseLogs()) {
			console.debug('[BinnedSearch] Empty priority level array, returning default values');
		}
		return {
			exclusiveLowerBound: -1,
			inclusiveUpperBound: 0,
			largestTokensSeen: 0
		};
	}

	// For very small arrays, simple binary search is faster
	if (sortedPriorityLevels.length < 16) {
		if (shouldPrintVerboseLogs()) {
			console.debug(`[BinnedSearch] Array too small (${sortedPriorityLevels.length} < 16), using standard binary search`);
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
	}

	// Create bins from the sorted priority levels
	const binSize = Math.max(5, Math.floor(sortedPriorityLevels.length / 10)); // Aim for about 10 bins, minimum size 5
	const bins = createBins(sortedPriorityLevels, binSize);

	if (shouldPrintVerboseLogs()) {
		console.debug(`[BinnedSearch] Created ${bins.length} bins with approximate size ${binSize}`);
		console.debug(`[BinnedSearch] Bin boundaries: ${bins.map(bin => bin[0]).join(', ')}`);
	}

	// Create an array of representatives from each bin (using the middle element)
	const representatives: number[] = bins.map(bin => bin[Math.floor(bin.length / 2)]);

	if (shouldPrintVerboseLogs()) {
		console.debug(`[BinnedSearch] Bin representatives: ${representatives.join(', ')}`);
	}

	// First, perform binary search on the representatives to find the right bin
	const binSearchResult = await runBinarySearch(
		elem,
		representatives,
		tokenizer,
		tokenLimit,
		usedTokenlimit,
		countTokensFast_UNSAFE,
		lastMessageIsIncomplete
	);

	largestTokenCountSeen = Math.max(largestTokenCountSeen, binSearchResult.largestTokensSeen);

	// Determine which bin to search in detail
	const targetBinIndex = binSearchResult.inclusiveUpperBound;

	if (shouldPrintVerboseLogs()) {
		console.debug(`[BinnedSearch] Binary search on representatives selected bin ${targetBinIndex}`);
	}

	// If no valid bin was found, use the most restrictive priority level
	if (targetBinIndex < 0 || targetBinIndex >= bins.length) {
		if (shouldPrintVerboseLogs()) {
			console.debug('[BinnedSearch] No valid bin found, using highest priority');
		}

		return {
			exclusiveLowerBound: -1,
			inclusiveUpperBound: 0,
			largestTokensSeen: largestTokenCountSeen
		};
	}

	// Get the target bin
	const targetBin = bins[targetBinIndex];

	if (shouldPrintVerboseLogs()) {
		console.debug(`[BinnedSearch] Searching within bin ${targetBinIndex}: ${targetBin.join(', ')}`);
	}

	// Perform detailed binary search within the selected bin
	const detailedResult = await runBinarySearch(
		elem,
		targetBin,
		tokenizer,
		tokenLimit,
		usedTokenlimit,
		countTokensFast_UNSAFE,
		lastMessageIsIncomplete
	);

	largestTokenCountSeen = Math.max(largestTokenCountSeen, detailedResult.largestTokensSeen);

	// Calculate the actual index in the original array
	const binStartIndex = targetBinIndex * binSize;
	const exclusiveLowerBound = detailedResult.exclusiveLowerBound === -1
		? -1
		: binStartIndex + detailedResult.exclusiveLowerBound;
	const inclusiveUpperBound = binStartIndex + detailedResult.inclusiveUpperBound;

	if (shouldPrintVerboseLogs()) {
		console.debug(`[BinnedSearch] Final result: exclusiveLowerBound=${exclusiveLowerBound}, inclusiveUpperBound=${inclusiveUpperBound}`);
	}

	return {
		exclusiveLowerBound,
		inclusiveUpperBound,
		largestTokensSeen: largestTokenCountSeen
	};
}