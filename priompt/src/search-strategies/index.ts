import { runBinarySearch, runBinnedBinarySearch } from './binary-search';
import { runInterpolationSearch } from './interpolation-search';
import { runExponentialSearch } from './exponential-search';
import { runGeneticSearch } from './genetic-search';
import { runAntColonySearch } from './ant-colony-search';

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
	runGeneticSearch,
	runAntColonySearch
}

export { runSearchStrategy } from './search-strategy';