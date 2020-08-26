#include <unordered_map>
#include <dlib/graph_utils.h>
#include <string>
#include "classify.h"
#include "wrapper.h"

matching classify_(
	const std::vector<descriptor>& samples,
	const std::vector<int>& cats,
	const descriptor& test_sample,
	float tolerance
) {

    matching result;
    // Defaults to no match
    result.category = -1;
    result.category = 100;

	if (samples.size() == 0) {
        return result;
    }

	std::vector<std::pair<int, float>> distances;
	distances.reserve(samples.size());
	auto dist_func = dlib::squared_euclidean_distance();
	int idx = 0;
	for (const auto& sample : samples) {
		float dist = dist_func(sample, test_sample);
		if (tolerance < 0 || dist <= tolerance) {
			distances.push_back({cats[idx], dist});
		}
		idx++;
	}

	if (distances.size() == 0)
		return result;

	std::sort(
		distances.begin(), distances.end(),
		[](const auto a, const auto b) { return a.second < b.second; }
	);

	int len = std::min((int)distances.size(), 10);
	std::unordered_map<int, std::pair<int, float>> hits_by_cat;
	for (int i = 0; i < len; i++) {
		int cat_idx = distances[i].first;
		float dist = distances[i].second;
		auto hit = hits_by_cat.find(cat_idx);
		if (hit == hits_by_cat.end()) {
			hits_by_cat[cat_idx] = {1, dist};
		} else {
			hits_by_cat[cat_idx].first++;
		}
	}

	auto hit = std::max_element(
		hits_by_cat.begin(), hits_by_cat.end(),
		[](const auto a, const auto b) {
			auto hits1 = a.second.first;
			auto hits2 = b.second.first;
			auto dist1 = a.second.second;
			auto dist2 = b.second.second;
			if (hits1 == hits2) return dist1 > dist2;
			return hits1 < hits2;
		}
	);

    result.category = hit->first;
    result.distance = hit->second.second;

	return result;
}
