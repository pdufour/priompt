import { runBinarySearch, runBinnedBinarySearch } from './binary-search';
import { runInterpolationSearch } from './interpolation-search';
import { runExponentialSearch } from './exponential-search';
import { runGeneticSearch } from './genetic-search';

export type SearchResult = {
	exclusiveLowerBound: number;
	inclusiveUpperBound: number;
	largestTokensSeen: number;
}

export {
	runBinarySearch,
	runInterpolationSearch,
	runExponentialSearch,
	runBinnedBinarySearch,
	runGeneticSearch
}

export { runSearchStrategy } from './search-strategy';