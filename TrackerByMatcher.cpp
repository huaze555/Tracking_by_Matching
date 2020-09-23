//
// Created by zengxiaojia on 20-5-6.
//
#include "DetectBox.h"
#include "TrackerByMatcher.hpp"

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
using namespace cv;
using namespace std;



namespace tbm {      ///  tracking by matching

/// ********************   kuhn_munkres.hpp     ***********************
    class KuhnMunkres {
    public:
        KuhnMunkres() : n_() {}

        ///
        /// \brief Solves the assignment problem for given dissimilarity matrix.
        /// It returns a vector that where each element is a column index for
        /// corresponding row (e.g. result[0] stores optimal column index for very
        /// first row in the dissimilarity matrix).
        /// \param dissimilarity_matrix CV_32F dissimilarity matrix.
        /// \return Optimal column index for each row. -1 means that there is no
        /// column for row.
        ///
        std::vector<size_t> Solve(const cv::Mat &dissimilarity_matrix) {
            CV_Assert(dissimilarity_matrix.type() == CV_32F);
            double min_val;
            cv::minMaxLoc(dissimilarity_matrix, &min_val);
            CV_Assert(min_val >= 0);

            n_ = std::max(dissimilarity_matrix.rows, dissimilarity_matrix.cols);
            dm_ = cv::Mat(n_, n_, CV_32F, cv::Scalar(0));
            marked_ = cv::Mat(n_, n_, CV_8S, cv::Scalar(0));
            points_ = std::vector<cv::Point>(n_ * 2);

            dissimilarity_matrix.copyTo(dm_(
                    cv::Rect(0, 0, dissimilarity_matrix.cols, dissimilarity_matrix.rows)));

            is_row_visited_ = std::vector<int>(n_, 0);
            is_col_visited_ = std::vector<int>(n_, 0);

            Run();

            std::vector<size_t> results(static_cast<size_t>(marked_.rows), static_cast<size_t>(-1));
            for (int i = 0; i < marked_.rows; i++) {
                const auto ptr = marked_.ptr<char>(i);
                for (int j = 0; j < marked_.cols; j++) {
                    if (ptr[j] == kStar) {
                        results[i] = j;
                    }
                }
            }
            return results;
        }

    private:
        static constexpr int kStar = 1;
        static constexpr int kPrime = 2;

        cv::Mat dm_;
        cv::Mat marked_;
        std::vector<cv::Point> points_;

        std::vector<int> is_row_visited_;
        std::vector<int> is_col_visited_;

        int n_;

        void TrySimpleCase() {
            auto is_row_visited = std::vector<int>(n_, 0);
            auto is_col_visited = std::vector<int>(n_, 0);

            for (int row = 0; row < n_; row++) {
                auto ptr = dm_.ptr<float>(row);
                auto marked_ptr = marked_.ptr<char>(row);
                auto min_val = *std::min_element(ptr, ptr + n_);
                for (int col = 0; col < n_; col++) {
                    ptr[col] -= min_val;
                    if (ptr[col] == 0 && !is_col_visited[col] && !is_row_visited[row]) {
                        marked_ptr[col] = kStar;
                        is_col_visited[col] = 1;
                        is_row_visited[row] = 1;
                    }
                }
            }
        }

        bool CheckIfOptimumIsFound() {
            int count = 0;
            for (int i = 0; i < n_; i++) {
                const auto marked_ptr = marked_.ptr<char>(i);
                for (int j = 0; j < n_; j++) {
                    if (marked_ptr[j] == kStar) {
                        is_col_visited_[j] = 1;
                        count++;
                    }
                }
            }

            return count >= n_;
        }

        cv::Point FindUncoveredMinValPos() {
            auto min_val = std::numeric_limits<float>::max();
            cv::Point min_val_pos(-1, -1);
            for (int i = 0; i < n_; i++) {
                if (!is_row_visited_[i]) {
                    auto dm_ptr = dm_.ptr<float>(i);
                    for (int j = 0; j < n_; j++) {
                        if (!is_col_visited_[j] && dm_ptr[j] < min_val) {
                            min_val = dm_ptr[j];
                            min_val_pos = cv::Point(j, i);
                        }
                    }
                }
            }
            return min_val_pos;
        }

        void UpdateDissimilarityMatrix(float val) {
            for (int i = 0; i < n_; i++) {
                auto dm_ptr = dm_.ptr<float>(i);
                for (int j = 0; j < n_; j++) {
                    if (is_row_visited_[i]) dm_ptr[j] += val;
                    if (!is_col_visited_[j]) dm_ptr[j] -= val;
                }
            }
        }

        int FindInRow(int row, int what) {
            for (int j = 0; j < n_; j++) {
                if (marked_.at<char>(row, j) == what) {
                    return j;
                }
            }
            return -1;
        }

        int FindInCol(int col, int what) {
            for (int i = 0; i < n_; i++) {
                if (marked_.at<char>(i, col) == what) {
                    return i;
                }
            }
            return -1;
        }

        void Run() {
            TrySimpleCase();
            while (!CheckIfOptimumIsFound()) {
                while (true) {
                    auto point = FindUncoveredMinValPos();
                    auto min_val = dm_.at<float>(point.y, point.x);
                    if (min_val > 0) {
                        UpdateDissimilarityMatrix(min_val);
                    } else {
                        marked_.at<char>(point.y, point.x) = kPrime;
                        int col = FindInRow(point.y, kStar);
                        if (col >= 0) {
                            is_row_visited_[point.y] = 1;
                            is_col_visited_[col] = 0;
                        } else {
                            int count = 0;
                            points_[count] = point;

                            while (true) {
                                int row = FindInCol(points_[count].x, kStar);
                                if (row >= 0) {
                                    count++;
                                    points_[count] = cv::Point(points_[count - 1].x, row);
                                    int col1 = FindInRow(points_[count].y, kPrime);
                                    count++;
                                    points_[count] = cv::Point(col1, points_[count - 1].y);
                                } else {
                                    break;
                                }
                            }

                            for (int i = 0; i < count + 1; i++) {
                                auto &mark = marked_.at<char>(points_[i].y, points_[i].x);
                                mark = mark == kStar ? 0 : kStar;
                            }

                            is_row_visited_ = std::vector<int>(n_, 0);
                            is_col_visited_ = std::vector<int>(n_, 0);

                            marked_.setTo(0, marked_ == kPrime);
                            break;
                        }
                    }
                }
            }
        }


    };///  end kuhn_munkres.hpp






    /// ****************** TrackingByMatcher  *******************

    CosDistance::CosDistance(const cv::Size &descriptor_size)
            : descriptor_size_(descriptor_size) {
        CV_Assert(descriptor_size.area() != 0);
    }

    float CosDistance::compute(const cv::Mat &descr1, const cv::Mat &descr2) {
        CV_Assert(!descr1.empty());
        CV_Assert(!descr2.empty());
        CV_Assert(descr1.size() == descriptor_size_);
        CV_Assert(descr2.size() == descriptor_size_);

        double xy = descr1.dot(descr2);
        double xx = descr1.dot(descr1);
        double yy = descr2.dot(descr2);
        double norm = sqrt(xx * yy) + 1e-6;
        return 0.5f * static_cast<float>(1.0 - xy / norm);
    }

    std::vector<float> CosDistance::compute(const std::vector<cv::Mat> &descrs1,
                                            const std::vector<cv::Mat> &descrs2) {
        CV_Assert(!descrs1.empty());
        CV_Assert(descrs1.size() == descrs2.size());

        std::vector<float> distances(descrs1.size(), 1.f);
        for (size_t i = 0; i < descrs1.size(); i++) {
            distances.at(i) = compute(descrs1.at(i), descrs2.at(i));
        }

        return distances;
    }


    float MatchTemplateDistance::compute(const cv::Mat &descr1,
                                         const cv::Mat &descr2) {
        CV_Assert(!descr1.empty() && !descr2.empty());
        CV_Assert(descr1.size() == descr2.size());
        CV_Assert(descr1.type() == descr2.type());
        cv::Mat res;
        cv::matchTemplate(descr1, descr2, res, type_);
        CV_Assert(res.size() == cv::Size(1, 1));
        float dist = res.at<float>(0, 0);
        return scale_ * dist + offset_;
    }

    std::vector<float> MatchTemplateDistance::compute(const std::vector<cv::Mat> &descrs1,
                                                      const std::vector<cv::Mat> &descrs2) {
        std::vector<float> result;
        for (size_t i = 0; i < descrs1.size(); i++) {
            result.push_back(compute(descrs1[i], descrs2[i]));
        }
        return result;
    }

    namespace {
        cv::Point Center(const cv::Rect &rect) {
            return cv::Point((int) (rect.x + rect.width * .5), (int) (rect.y + rect.height * .5));
        }

        std::vector<cv::Point> Centers(const TrackedObjects &detections) {
            std::vector<cv::Point> centers(detections.size());
            for (size_t i = 0; i < detections.size(); i++) {
                centers[i] = Center(detections[i].rect);
            }
            return centers;
        }

        inline bool IsInRange(float val, float min, float max) {
            return min <= val && val <= max;
        }

        inline bool IsInRange(float val, cv::Vec2f range) {
            return IsInRange(val, range[0], range[1]);
        }

        std::vector<cv::Scalar> GenRandomColors(int colors_num) {
            std::vector<cv::Scalar> colors(colors_num);
            for (int i = 0; i < colors_num; i++) {
                colors[i] = cv::Scalar(static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                                       static_cast<uchar>(255. * rand() / RAND_MAX),  // NOLINT
                                       static_cast<uchar>(255. * rand() / RAND_MAX));  // NOLINT
            }
            return colors;
        }

        ///
        /// \brief Draws a polyline on a frame.
        /// \param[in] polyline Vector of points (polyline).
        /// \param[in] color Color (BGR).
        /// \param[in,out] image Frame.
        /// \param[in] lwd Line width.
        ///
        void DrawPolyline(const std::vector<cv::Point> &polyline,
                          const cv::Scalar &color, CV_OUT cv::Mat &image,
                          int lwd = 5) {
            CV_Assert(!image.empty());
            CV_Assert(image.type() == CV_8UC3);
            CV_Assert(lwd > 0);

            for (size_t i = 1; i < polyline.size(); i++) {
                cv::line(image, polyline[i - 1], polyline[i], color, lwd);
            }
        }

        void ValidateParams(const TrackerParams &p) {
            CV_Assert(p.min_track_duration >= static_cast<size_t>(500));
            CV_Assert(p.min_track_duration <= static_cast<size_t>(10000));

            CV_Assert(p.forget_delay <= static_cast<size_t>(10000));

            CV_Assert(p.aff_thr_fast >= 0.0f);
            CV_Assert(p.aff_thr_fast <= 1.0f);

            CV_Assert(p.aff_thr_strong >= 0.0f);
            CV_Assert(p.aff_thr_strong <= 1.0f);

            CV_Assert(p.shape_affinity_w >= 0.0f);
            CV_Assert(p.shape_affinity_w <= 100.0f);

            CV_Assert(p.motion_affinity_w >= 0.0f);
            CV_Assert(p.motion_affinity_w <= 100.0f);

            CV_Assert(p.time_affinity_w >= 0.0f);
            CV_Assert(p.time_affinity_w <= 100.0f);

            CV_Assert(p.min_det_conf >= 0.0f);
            CV_Assert(p.min_det_conf <= 1.0f);

            CV_Assert(p.bbox_aspect_ratios_range[0] >= 0.0f);
            CV_Assert(p.bbox_aspect_ratios_range[1] <= 10.0f);
            CV_Assert(p.bbox_aspect_ratios_range[0] < p.bbox_aspect_ratios_range[1]);

            CV_Assert(p.bbox_heights_range[0] >= 10.0f);
            CV_Assert(p.bbox_heights_range[1] <= 1080.0f);
            CV_Assert(p.bbox_heights_range[0] < p.bbox_heights_range[1]);

            CV_Assert(p.predict >= 0);
            CV_Assert(p.predict <= 10000);

            CV_Assert(p.strong_affinity_thr >= 0.0f);
            CV_Assert(p.strong_affinity_thr <= 1.0f);

            CV_Assert(p.reid_thr >= 0.0f);
            CV_Assert(p.reid_thr <= 1.0f);


            if (p.max_num_objects_in_track > 0) {
                int min_required_track_length = static_cast<int>(p.forget_delay);
                CV_Assert(p.max_num_objects_in_track >= min_required_track_length);
                CV_Assert(p.max_num_objects_in_track <= 10000);
            }
        }

    }  // anonymous namespace



    cv::Ptr<ITrackerByMatching> createTrackerByMatching(const TrackerParams &params) {
        ITrackerByMatching *ptr = new TrackerByMatching(params);
        return cv::Ptr<ITrackerByMatching>(ptr);
    }

    TrackerParams::TrackerParams()
            : min_track_duration(1000),
              forget_delay(150),
              aff_thr_fast(0.8f),
              aff_thr_strong(0.75f),
              shape_affinity_w(0.5f),
              motion_affinity_w(0.2f),
              time_affinity_w(0.0f),
              min_det_conf(0.1f),
              bbox_aspect_ratios_range(0.666f, 5.0f),
              bbox_heights_range(40.f, 1000.f),
              predict(25),
              strong_affinity_thr(0.2805f),
              reid_thr(0.61f),
              drop_forgotten_tracks(true),
              max_num_objects_in_track(300) {}

// Returns confusion matrix as:
//   |tp fn|
//   |fp tn|
    cv::Mat TrackerByMatching::ConfusionMatrix(const std::vector<Match> &matches) {
        const bool kNegative = false;
        cv::Mat conf_mat(2, 2, CV_32F, cv::Scalar(0));
        for (const auto &m : matches) {
            conf_mat.at<float>(m.gt_label == kNegative, m.pr_label == kNegative)++;
        }

        return conf_mat;
    }

    TrackerByMatching::TrackerByMatching(const TrackerParams &params)
            : params_(params),
              descriptor_strong_(nullptr),
              distance_strong_(nullptr),
              collect_matches_(true),
              tracks_counter_(0),
              valid_tracks_counter_(0),
              frame_size_(0, 0),
              prev_timestamp_(std::numeric_limits<uint64_t>::max()) {
        ValidateParams(params);
    }

// Pipeline parameters getter.
    const TrackerParams &TrackerByMatching::params() const { return params_; }

// Pipeline parameters setter.
    void TrackerByMatching::setParams(const TrackerParams &params) {
        ValidateParams(params);
        params_ = params;
    }

// Descriptor fast getter.
    const TrackerByMatching::Descriptor &TrackerByMatching::descriptorFast() const {
        return descriptor_fast_;
    }

// Descriptor fast setter.
    void TrackerByMatching::setDescriptorFast(const Descriptor &val) {
        descriptor_fast_ = val;
    }

// Descriptor strong getter.
    const TrackerByMatching::Descriptor &TrackerByMatching::descriptorStrong() const {
        return descriptor_strong_;
    }

// Descriptor strong setter.
    void TrackerByMatching::setDescriptorStrong(const Descriptor &val) {
        descriptor_strong_ = val;
    }

// Distance fast getter.
    const TrackerByMatching::Distance &TrackerByMatching::distanceFast() const { return distance_fast_; }

// Distance fast setter.
    void TrackerByMatching::setDistanceFast(const Distance &val) { distance_fast_ = val; }

// Distance strong getter.
    const TrackerByMatching::Distance &TrackerByMatching::distanceStrong() const { return distance_strong_; }

// Distance strong setter.
    void TrackerByMatching::setDistanceStrong(const Distance &val) { distance_strong_ = val; }

// Returns all tracks including forgotten (lost too many frames ago).
    const std::unordered_map<size_t, Track> &
    TrackerByMatching::tracks() const {
        return tracks_;
    }

// Returns indexes of active tracks only.
    const std::set<size_t> &TrackerByMatching::active_track_ids() const {
        return active_track_ids_;
    }


// Returns decisions made by heuristic based on fast distance/descriptor and
// shape, motion and time affinity.
    const std::vector<TrackerByMatching::Match> &
    TrackerByMatching::base_classifier_matches() const {
        return base_classifier_matches_;
    }

// Returns decisions made by heuristic based on strong distance/descriptor
// and
// shape, motion and time affinity.
    const std::vector<TrackerByMatching::Match> &TrackerByMatching::reid_based_classifier_matches() const {
        return reid_based_classifier_matches_;
    }

// Returns decisions made by strong distance/descriptor affinity.
    const std::vector<TrackerByMatching::Match> &TrackerByMatching::reid_classifier_matches() const {
        return reid_classifier_matches_;
    }

    TrackedObjects TrackerByMatching::FilterDetections(const TrackedObjects &detections) const {
        TrackedObjects filtered_detections;
        for (const auto &det : detections) {
            float aspect_ratio = static_cast<float>(det.rect.height) / det.rect.width;
            if (det.confidence > params_.min_det_conf &&
                IsInRange(aspect_ratio, params_.bbox_aspect_ratios_range) &&
                IsInRange(static_cast<float>(det.rect.height), params_.bbox_heights_range)) {
                filtered_detections.emplace_back(det);
            }
        }
        return filtered_detections;
    }

    void TrackerByMatching::SolveAssignmentProblem(
            const std::set<size_t> &track_ids, const TrackedObjects &detections,
            const std::vector<cv::Mat> &descriptors,
            std::set<size_t> &unmatched_tracks, std::set<size_t> &unmatched_detections,
            std::set<std::tuple<size_t, size_t, float>> &matches) {
        unmatched_tracks.clear();
        unmatched_detections.clear();

        CV_Assert(!track_ids.empty());
        CV_Assert(!detections.empty());
        CV_Assert(descriptors.size() == detections.size());
        matches.clear();

        cv::Mat dissimilarity;
        ComputeDissimilarityMatrix(track_ids, detections, descriptors,
                                   dissimilarity);

        auto res = KuhnMunkres().Solve(dissimilarity);

        for (size_t i = 0; i < detections.size(); i++) {
            unmatched_detections.insert(i);
        }

        int i = 0;
        for (auto id : track_ids) {
            if (res[i] < detections.size()) {
                matches.emplace(id, res[i], 1 - dissimilarity.at<float>(i, static_cast<int>(res[i])));
            } else {
                unmatched_tracks.insert(id);
            }
            i++;
        }
    }

    const ObjectTracks TrackerByMatching::all_tracks(bool valid_only) const {
        ObjectTracks all_objects;
        int counter = 0;

        std::set<size_t> sorted_ids;
        for (const auto &pair : tracks()) {
            sorted_ids.emplace(pair.first);
        }

        for (size_t id : sorted_ids) {
            if (!valid_only || isTrackValid(id)) {
                TrackedObjects filtered_objects;
                for (const auto &object : tracks().at(id).objects) {
                    filtered_objects.emplace_back(object);
                    filtered_objects.back().object_id = counter;
                }
                all_objects.emplace(counter++, filtered_objects);
            }
        }
        return all_objects;
    }

    cv::Rect TrackerByMatching::PredictRect(size_t id, size_t k,
                                            size_t s) const {
        const auto &track = tracks_.at(id);
        CV_Assert(!track.empty());

        if (track.size() == 1) {
            return track[0].rect;
        }

        size_t start_i = track.size() > k ? track.size() - k : 0;
        float width = 0, height = 0;

        for (size_t i = start_i; i < track.size(); i++) {
            width += track[i].rect.width;
            height += track[i].rect.height;
        }

        CV_Assert(track.size() - start_i > 0);
        width /= (track.size() - start_i);
        height /= (track.size() - start_i);

        float delim = 0;
        cv::Point2f d(0, 0);

        for (size_t i = start_i + 1; i < track.size(); i++) {
            d += cv::Point2f(Center(track[i].rect) - Center(track[i - 1].rect));
            delim += (track[i].frame_idx - track[i - 1].frame_idx);
        }

        if (delim) {
            d /= delim;
        }

        s += 1;

        cv::Point c = Center(track.back().rect);
        return cv::Rect(static_cast<int>(c.x - width / 2 + d.x * s),
                        static_cast<int>(c.y - height / 2 + d.y * s),
                        static_cast<int>(width),
                        static_cast<int>(height));
    }


    bool TrackerByMatching::EraseTrackIfBBoxIsOutOfFrame(size_t track_id) {
        if (tracks_.find(track_id) == tracks_.end()) return true;
        auto c = Center(tracks_.at(track_id).predicted_rect);
        if (!prev_frame_size_.empty() &&
            (c.x < 0 || c.y < 0 || c.x > prev_frame_size_.width ||
             c.y > prev_frame_size_.height)) {
            tracks_.at(track_id).lost = params_.forget_delay + 1;
            for (auto id : active_track_ids()) {
                size_t min_id = std::min(id, track_id);
                size_t max_id = std::max(id, track_id);
                tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
            }
            active_track_ids_.erase(track_id);
            return true;
        }
        return false;
    }

    bool TrackerByMatching::EraseTrackIfItWasLostTooManyFramesAgo(
            size_t track_id) {
        if (tracks_.find(track_id) == tracks_.end()) return true;
        if (tracks_.at(track_id).lost > params_.forget_delay) {
            for (auto id : active_track_ids()) {
                size_t min_id = std::min(id, track_id);
                size_t max_id = std::max(id, track_id);
                tracks_dists_.erase(std::pair<size_t, size_t>(min_id, max_id));
            }
            active_track_ids_.erase(track_id);

            return true;
        }
        return false;
    }

    bool TrackerByMatching::UpdateLostTrackAndEraseIfItsNeeded(
            size_t track_id) {
        tracks_.at(track_id).lost++;
        tracks_.at(track_id).predicted_rect =
                PredictRect(track_id, params().predict, tracks_.at(track_id).lost);

        bool erased = EraseTrackIfBBoxIsOutOfFrame(track_id);
        if (!erased) erased = EraseTrackIfItWasLostTooManyFramesAgo(track_id);
        return erased;
    }

    void TrackerByMatching::UpdateLostTracks(
            const std::set<size_t> &track_ids) {
        for (auto track_id : track_ids) {
            UpdateLostTrackAndEraseIfItsNeeded(track_id);
        }
    }

    void TrackerByMatching::process(const cv::Mat &frame,
                                    const TrackedObjects &input_detections,
                                    uint64_t timestamp) {
        if (prev_timestamp_ != std::numeric_limits<uint64_t>::max())
            CV_Assert(static_cast<size_t>(prev_timestamp_) < static_cast<size_t>(timestamp));

        if (frame_size_ == cv::Size(0, 0)) {
            frame_size_ = frame.size();
        } else {
            CV_Assert(frame_size_ == frame.size());
        }

        TrackedObjects detections = FilterDetections(input_detections);
        for (auto &obj : detections) {
            obj.timestamp = timestamp;
        }

        std::vector<cv::Mat> descriptors_fast;
        ComputeFastDesciptors(frame, detections, descriptors_fast);


        auto active_tracks = active_track_ids_;

        if (!active_tracks.empty() && !detections.empty()) {
            std::set<size_t> unmatched_tracks, unmatched_detections;
            std::set<std::tuple<size_t, size_t, float>> matches;    //  {track_id, detecton_id, score}

            SolveAssignmentProblem(active_tracks, detections, descriptors_fast,
                                   unmatched_tracks,
                                   unmatched_detections, matches);

            std::map<size_t, std::pair<bool, cv::Mat>> is_matching_to_track;

            if (distance_strong_) {
                std::vector<std::pair<size_t, size_t>> reid_track_and_det_ids =
                        GetTrackToDetectionIds(matches);
                is_matching_to_track = StrongMatching(
                        frame, detections, reid_track_and_det_ids);
            }

            for (const auto &match : matches) {
                size_t track_id = std::get<0>(match);
                size_t det_id = std::get<1>(match);
                float conf = std::get<2>(match);

                auto last_det = tracks_.at(track_id).objects.back();
                last_det.rect = tracks_.at(track_id).predicted_rect;

                if (collect_matches_ && last_det.object_id >= 0 &&
                    detections[det_id].object_id >= 0) {
                    base_classifier_matches_.emplace_back(
                            tracks_.at(track_id).objects.back(), last_det.rect,
                            detections[det_id], conf > params_.aff_thr_fast);
                }

                if (conf > params_.aff_thr_fast) {
                    AppendToTrack(frame, track_id, detections[det_id],
                                  descriptors_fast[det_id], cv::Mat());
                    unmatched_detections.erase(det_id);
                } else {
                    if (conf > params_.strong_affinity_thr) {
                        if (distance_strong_ && is_matching_to_track[track_id].first) {
                            AppendToTrack(frame, track_id, detections[det_id],
                                          descriptors_fast[det_id],
                                          is_matching_to_track[track_id].second.clone());
                        } else {
                            if (UpdateLostTrackAndEraseIfItsNeeded(track_id)) {
                                AddNewTrack(frame, detections[det_id], descriptors_fast[det_id],
                                            distance_strong_
                                            ? is_matching_to_track[track_id].second.clone()
                                            : cv::Mat());
                            }
                        }

                        unmatched_detections.erase(det_id);
                    } else {
                        unmatched_tracks.insert(track_id);
                    }
                }
            }

            AddNewTracks(frame, detections, descriptors_fast, unmatched_detections);
            UpdateLostTracks(unmatched_tracks);

            for (size_t id : active_tracks) {
                EraseTrackIfBBoxIsOutOfFrame(id);
            }
        } else {
            AddNewTracks(frame, detections, descriptors_fast);
            UpdateLostTracks(active_tracks);
        }

        prev_frame_size_ = frame.size();
        if (params_.drop_forgotten_tracks) dropForgottenTracks();

        tracks_dists_.clear();
        prev_timestamp_ = timestamp;
    }

    void TrackerByMatching::dropForgottenTracks() {
        std::unordered_map<size_t, Track> new_tracks;
        std::set<size_t> new_active_tracks;

        size_t max_id = 0;
        if (!active_track_ids_.empty())
            max_id =
                    *std::max_element(active_track_ids_.begin(), active_track_ids_.end());

        const size_t kMaxTrackID = 10000;
        bool reassign_id = max_id > kMaxTrackID;

        size_t counter = 0;
        for (const auto &pair : tracks_) {
            if (!isTrackForgotten(pair.first)) {
                new_tracks.emplace(reassign_id ? counter : pair.first, pair.second);
                new_active_tracks.emplace(reassign_id ? counter : pair.first);
                counter++;

            } else {
                if (isTrackValid(pair.first)) {
                    valid_tracks_counter_++;
                }
            }
        }
        tracks_.swap(new_tracks);
        active_track_ids_.swap(new_active_tracks);

        tracks_counter_ = reassign_id ? counter : tracks_counter_;
    }

    void TrackerByMatching::dropForgottenTrack(size_t track_id) {
        CV_Assert(isTrackForgotten(track_id));
        CV_Assert(active_track_ids_.count(track_id) == 0);
        tracks_.erase(track_id);
    }

    float TrackerByMatching::ShapeAffinity(float weight, const cv::Rect &trk,
                                           const cv::Rect &det) {
        float w_dist = static_cast<float>(std::fabs(trk.width - det.width) / (trk.width + det.width));
        float h_dist = static_cast<float>(std::fabs(trk.height - det.height) / (trk.height + det.height));
        return exp(-weight * (w_dist + h_dist));
    }

    float TrackerByMatching::MotionAffinity(float weight, const cv::Rect &trk,
                                            const cv::Rect &det) {
        float x_dist = static_cast<float>(trk.x - det.x) * (trk.x - det.x) /
                       (det.width * det.width);
        float y_dist = static_cast<float>(trk.y - det.y) * (trk.y - det.y) /
                       (det.height * det.height);
        return exp(-weight * (x_dist + y_dist));
    }

    float TrackerByMatching::TimeAffinity(float weight, const float &trk_time,
                                          const float &det_time) {
        return exp(-weight * std::fabs(trk_time - det_time));
    }

    void TrackerByMatching::ComputeFastDesciptors(
            const cv::Mat &frame, const TrackedObjects &detections,
            std::vector<cv::Mat> &desriptors) {
        desriptors = std::vector<cv::Mat>(detections.size(), cv::Mat());
        for (size_t i = 0; i < detections.size(); i++) {
            descriptor_fast_->compute(frame(detections[i].rect).clone(),
                                      desriptors[i]);
        }
    }

    void TrackerByMatching::ComputeDissimilarityMatrix(
            const std::set<size_t> &active_tracks, const TrackedObjects &detections,
            const std::vector<cv::Mat> &descriptors_fast,
            cv::Mat &dissimilarity_matrix) {
        cv::Mat am(static_cast<int>(active_tracks.size()), static_cast<int>(detections.size()), CV_32F, cv::Scalar(0));
        int i = 0;
        for (auto id : active_tracks) {
            auto ptr = am.ptr<float>(i);
            for (size_t j = 0; j < descriptors_fast.size(); j++) {
                auto last_det = tracks_.at(id).objects.back();
                last_det.rect = tracks_.at(id).predicted_rect;
                ptr[j] = AffinityFast(tracks_.at(id).descriptor_fast, last_det,
                                      descriptors_fast[j], detections[j]);
            }
            i++;
        }
        dissimilarity_matrix = 1.0 - am;
    }

    std::vector<float> TrackerByMatching::ComputeDistances(
            const cv::Mat &frame,
            const TrackedObjects &detections,
            const std::vector<std::pair<size_t, size_t>> &track_and_det_ids,
            std::map<size_t, cv::Mat> &det_id_to_descriptor) {
        std::map<size_t, size_t> det_to_batch_ids;
        std::map<size_t, size_t> track_to_batch_ids;

        std::vector<cv::Mat> images;
        std::vector<cv::Mat> descriptors;
        for (size_t i = 0; i < track_and_det_ids.size(); i++) {
            size_t track_id = track_and_det_ids[i].first;
            size_t det_id = track_and_det_ids[i].second;

            if (tracks_.at(track_id).descriptor_strong.empty()) {
                images.push_back(tracks_.at(track_id).last_image);
                descriptors.push_back(cv::Mat());
                track_to_batch_ids[track_id] = descriptors.size() - 1;
            }

            images.push_back(frame(detections[det_id].rect));
            descriptors.push_back(cv::Mat());
            det_to_batch_ids[det_id] = descriptors.size() - 1;
        }

        descriptor_strong_->compute(images, descriptors);

        std::vector<cv::Mat> descriptors1;
        std::vector<cv::Mat> descriptors2;
        for (size_t i = 0; i < track_and_det_ids.size(); i++) {
            size_t track_id = track_and_det_ids[i].first;
            size_t det_id = track_and_det_ids[i].second;

            if (tracks_.at(track_id).descriptor_strong.empty()) {
                tracks_.at(track_id).descriptor_strong =
                        descriptors[track_to_batch_ids[track_id]].clone();
            }
            det_id_to_descriptor[det_id] = descriptors[det_to_batch_ids[det_id]];

            descriptors1.push_back(descriptors[det_to_batch_ids[det_id]]);
            descriptors2.push_back(tracks_.at(track_id).descriptor_strong);
        }

        std::vector<float> distances =
                distance_strong_->compute(descriptors1, descriptors2);

        return distances;
    }

    std::vector<std::pair<size_t, size_t>>
    TrackerByMatching::GetTrackToDetectionIds(
            const std::set<std::tuple<size_t, size_t, float>> &matches) {
        std::vector<std::pair<size_t, size_t>> track_and_det_ids;

        for (const auto &match : matches) {
            size_t track_id = std::get<0>(match);
            size_t det_id = std::get<1>(match);
            float conf = std::get<2>(match);
            if (conf < params_.aff_thr_fast && conf > params_.strong_affinity_thr) {
                track_and_det_ids.emplace_back(track_id, det_id);
            }
        }
        return track_and_det_ids;
    }

    std::map<size_t, std::pair<bool, cv::Mat>>
    TrackerByMatching::StrongMatching(
            const cv::Mat &frame,
            const TrackedObjects &detections,
            const std::vector<std::pair<size_t, size_t>> &track_and_det_ids) {
        std::map<size_t, std::pair<bool, cv::Mat>> is_matching;

        if (track_and_det_ids.size() == 0) {
            return is_matching;
        }

        std::map<size_t, cv::Mat> det_ids_to_descriptors;
        std::vector<float> distances =
                ComputeDistances(frame, detections,
                                 track_and_det_ids, det_ids_to_descriptors);

        for (size_t i = 0; i < track_and_det_ids.size(); i++) {
            auto reid_affinity = 1.0 - distances[i];

            size_t track_id = track_and_det_ids[i].first;
            size_t det_id = track_and_det_ids[i].second;

            const auto &track = tracks_.at(track_id);
            const auto &detection = detections[det_id];

            auto last_det = track.objects.back();
            last_det.rect = track.predicted_rect;

            float affinity = static_cast<float>(reid_affinity * Affinity(last_det, detection));

            if (collect_matches_ && last_det.object_id >= 0 &&
                detection.object_id >= 0) {
                reid_classifier_matches_.emplace_back(track.objects.back(), last_det.rect,
                                                      detection,
                                                      reid_affinity > params_.reid_thr);

                reid_based_classifier_matches_.emplace_back(
                        track.objects.back(), last_det.rect, detection,
                        affinity > params_.aff_thr_strong);
            }

            bool is_detection_matching =
                    reid_affinity > params_.reid_thr && affinity > params_.aff_thr_strong;

            is_matching[track_id] = std::pair<bool, cv::Mat>(
                    is_detection_matching, det_ids_to_descriptors[det_id]);
        }
        return is_matching;
    }

    void TrackerByMatching::AddNewTracks(const cv::Mat &frame, const TrackedObjects &detections, const std::vector<cv::Mat> &descriptors_fast) {
        CV_Assert(detections.size() == descriptors_fast.size());
        for (size_t i = 0; i < detections.size(); i++) {
            AddNewTrack(frame, detections[i], descriptors_fast[i]);
        }
    }

    void TrackerByMatching::AddNewTracks(
            const cv::Mat &frame, const TrackedObjects &detections,
            const std::vector<cv::Mat> &descriptors_fast, const std::set<size_t> &ids) {
        CV_Assert(detections.size() == descriptors_fast.size());
        for (size_t i : ids) {
            CV_Assert(i < detections.size());
            AddNewTrack(frame, detections[i], descriptors_fast[i]);
        }
    }

    void TrackerByMatching::AddNewTrack(const cv::Mat &frame,
                                        const TrackedObject &detection,
                                        const cv::Mat &descriptor_fast,
                                        const cv::Mat &descriptor_strong) {
        auto detection_with_id = detection;
        detection_with_id.object_id = static_cast<int>(tracks_counter_);
        tracks_.emplace(std::pair<size_t, Track>(tracks_counter_,
                Track({detection_with_id}, frame(detection.rect).clone(),
                      descriptor_fast.clone(), descriptor_strong.clone())));

        for (size_t id : active_track_ids_) {
            tracks_dists_.emplace(std::pair<size_t, size_t>(id, tracks_counter_),
                                  std::numeric_limits<float>::max());
        }

        active_track_ids_.insert(tracks_counter_);
        tracks_counter_++;
    }

    void TrackerByMatching::AppendToTrack(const cv::Mat &frame,
                                          size_t track_id,
                                          const TrackedObject &detection,
                                          const cv::Mat &descriptor_fast,
                                          const cv::Mat &descriptor_strong) {
        CV_Assert(!isTrackForgotten(track_id));

        auto detection_with_id = detection;
        detection_with_id.object_id = static_cast<int>(track_id);

        auto &cur_track = tracks_.at(track_id);
        cur_track.objects.emplace_back(detection_with_id);
        cur_track.predicted_rect = detection.rect;
        cur_track.lost = 0;
        cur_track.last_image = frame(detection.rect).clone();
        cur_track.descriptor_fast = descriptor_fast.clone();
        cur_track.length++;

        if (cur_track.descriptor_strong.empty()) {
            cur_track.descriptor_strong = descriptor_strong.clone();
        } else if (!descriptor_strong.empty()) {
            cur_track.descriptor_strong =
                    0.5 * (descriptor_strong + cur_track.descriptor_strong);
        }


        if (params_.max_num_objects_in_track > 0) {
            while (cur_track.size() >
                   static_cast<size_t>(params_.max_num_objects_in_track)) {
                cur_track.objects.erase(cur_track.objects.begin());
            }
        }
    }

    float TrackerByMatching::AffinityFast(const cv::Mat &descriptor1,
                                          const TrackedObject &obj1,
                                          const cv::Mat &descriptor2,
                                          const TrackedObject &obj2) {
        const float eps = static_cast<float>(1e-6);
        float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
        if (shp_aff < eps) return 0.0;

        float mot_aff =
                MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
        if (mot_aff < eps) return 0.0;
        float time_aff =
                TimeAffinity(params_.time_affinity_w, static_cast<float>(obj1.frame_idx),
                             static_cast<float>(obj2.frame_idx));

        if (time_aff < eps) return 0.0;

        float app_aff = static_cast<float>(1.0 - distance_fast_->compute(descriptor1, descriptor2));

        return shp_aff * mot_aff * app_aff * time_aff;
    }

    float TrackerByMatching::Affinity(const TrackedObject &obj1,
                                      const TrackedObject &obj2) {
        float shp_aff = ShapeAffinity(params_.shape_affinity_w, obj1.rect, obj2.rect);
        float mot_aff =
                MotionAffinity(params_.motion_affinity_w, obj1.rect, obj2.rect);
        float time_aff =
                TimeAffinity(params_.time_affinity_w, static_cast<float>(obj1.frame_idx),
                             static_cast<float>(obj2.frame_idx));
        return shp_aff * mot_aff * time_aff;
    }

    bool TrackerByMatching::isTrackValid(size_t id) const {
        const auto &track = tracks_.at(id);
        const auto &objects = track.objects;
        if (objects.empty()) {
            return false;
        }
        int64_t duration_ms = objects.back().timestamp - track.first_object.timestamp;
        if (duration_ms < static_cast<int64_t>(params_.min_track_duration))
            return false;
        return true;
    }

    bool TrackerByMatching::isTrackForgotten(size_t id) const {
        return isTrackForgotten(tracks_.at(id));
    }

    bool TrackerByMatching::isTrackForgotten(const Track &track) const {
        return (track.lost > params_.forget_delay);
    }

    size_t TrackerByMatching::count() const {
        size_t count = valid_tracks_counter_;
        for (const auto &pair : tracks_) {
            count += (isTrackValid(pair.first) ? 1 : 0);
        }
        return count;
    }

    std::unordered_map<size_t, std::vector<cv::Point>>
    TrackerByMatching::getActiveTracks() const {
        std::unordered_map<size_t, std::vector<cv::Point>> active_tracks;
        for (size_t idx : active_track_ids()) {
            auto track = tracks().at(idx);
            if (isTrackValid(idx) && !isTrackForgotten(idx)) {
                active_tracks.emplace(idx, Centers(track.objects));
            }
        }
        return active_tracks;
    }

    TrackedObjects TrackerByMatching::trackedDetections() const {
        TrackedObjects detections;
        for (size_t idx : active_track_ids()) {
            auto track = tracks().at(idx);
            if (isTrackValid(idx) && !track.lost) {
                detections.emplace_back(track.objects.back());
            }
        }
        return detections;
    }

    cv::Mat TrackerByMatching::drawActiveTracks(const cv::Mat &frame) {
        cv::Mat out_frame = frame.clone();

        if (colors_.empty()) {
            int num_colors = 100;
            colors_ = GenRandomColors(num_colors);
        }

        auto active_tracks = getActiveTracks();
        for (auto active_track : active_tracks) {
            size_t idx = active_track.first;
            auto centers = active_track.second;
            DrawPolyline(centers, colors_[idx % colors_.size()], out_frame);
            std::stringstream ss;
            ss << idx;
            cv::putText(out_frame, ss.str(), centers.back(), cv::FONT_HERSHEY_SCRIPT_COMPLEX, 2.0,
                        colors_[idx % colors_.size()], 3);
            auto track = tracks().at(idx);
            if (track.lost) {
                cv::line(out_frame, active_track.second.back(),
                         Center(track.predicted_rect), cv::Scalar(0, 0, 0), 4);
            }
        }

        return out_frame;
    }

    const cv::Size kMinFrameSize = cv::Size(320, 240);
    const cv::Size kMaxFrameSize = cv::Size(1920, 1080);

    void TrackerByMatching::PrintConfusionMatrices() const {
        std::cout << "Base classifier quality: " << std::endl;
        {
            auto cm = ConfusionMatrix(base_classifier_matches());
            std::cout << cm << std::endl;
            std::cout << "or" << std::endl;
            cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
            cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
            std::cout << cm << std::endl << std::endl;
        }

        std::cout << "Reid-based classifier quality: " << std::endl;
        {
            auto cm = ConfusionMatrix(reid_based_classifier_matches());
            std::cout << cm << std::endl;
            std::cout << "or" << std::endl;
            cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
            cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
            std::cout << cm << std::endl << std::endl;
        }

        std::cout << "Reid only classifier quality: " << std::endl;
        {
            auto cm = ConfusionMatrix(reid_classifier_matches());
            std::cout << cm << std::endl;
            std::cout << "or" << std::endl;
            cm.row(0) = cm.row(0) / std::max(1.0, cv::sum(cm.row(0))[0]);
            cm.row(1) = cm.row(1) / std::max(1.0, cv::sum(cm.row(1))[0]);
            std::cout << cm << std::endl << std::endl;
        }
    }




    /// Our Tracker Interface

// ***********************************************************
    TrackerByMatcher::TrackerByMatcher(float fps, const std::vector<int> &desired_class_id)
            : desired_class_id(desired_class_id), video_fps(fps) {

        //  TrackerParams
        //min_track_duration(1000),  最小跟踪时间 / ms
        //forget_delay(150),      如果最后检测到的box框超过该帧数，则遗忘该轨迹
        //aff_thr_fast(0.8f),     用于判断小段轨迹和检测框是否关联的亲和力阈值（快描述子使用）, 值越小，则越容易匹配，不容易跟丢，但可能switch id
        //aff_thr_strong(0.75f),   用于判断小段轨迹和检测框是否关联的亲和力阈值（强描述子使用）
        //shape_affinity_w(0.5f),   形状亲和力权重
        //motion_affinity_w(0.2f),   运动亲和力权重
        //time_affinity_w(0.0f),     时间亲和力权重
        //min_det_conf(0.1f),          检测框得最小置信度
        //bbox_aspect_ratios_range(0.666f, 5.0f),          检测框的宽高比比例范围
        //bbox_heights_range(40.f, 1000.f),                 检测框的高度范围
        //predict(25),                                      使用多少帧预测目标框，防止丢失轨迹
        //strong_affinity_thr(0.2805f),            如果“快”置信度大于此阈值，则使用“强” re-id 方法。
        //reid_thr(0.61f),                        re-id 的亲和力阈值
        //drop_forgotten_tracks(true),
        //max_num_objects_in_track(300)           最大跟踪数

        TrackerParams params;

        params.bbox_aspect_ratios_range = Vec2f(0.2f, 5.0f);
        params.bbox_heights_range = Vec2f(10.f, 1080.f);
        //params.min_track_duration = 500;
        //params.forget_delay = 50;
        params.aff_thr_fast = 0.5;
        params.predict = 50;
        params.max_num_objects_in_track = 1000;

        this->tracker = createTrackerByMatching(params);

        std::shared_ptr<IImageDescriptor> descriptor_fast = std::make_shared<ResizedImageDescriptor>(cv::Size(16, 32),
                                                                                                     cv::InterpolationFlags::INTER_LINEAR);
        std::shared_ptr<IDescriptorDistance> distance_fast = std::make_shared<CosDistance>(cv::Size(16, 32));
        tracker->setDescriptorFast(descriptor_fast);
        tracker->setDistanceFast(distance_fast);

        std::shared_ptr<IImageDescriptor> descriptor_strong = std::make_shared<ResizedImageDescriptor>(cv::Size(32, 32),
                                                                                                       cv::InterpolationFlags::INTER_LINEAR);
        std::shared_ptr<IDescriptorDistance> distance_strong = std::make_shared<CosDistance>(cv::Size(32, 32));
        tracker->setDescriptorStrong(descriptor_strong);
        tracker->setDistanceStrong(distance_strong);

    }

    TrackedObjects
    TrackerByMatcher::nms(const cv::Mat &frame, const std::vector<DetectBox> &bboxInfo, int frame_idx) {

        TrackedObjects valid_objs;

        // 对所有检测到的目标框，先过滤一遍
        for (const DetectBox &boxInfo : bboxInfo) {
            int class_id = boxInfo.class_id;
            if (!desired_class_id.empty() &&
                find(desired_class_id.begin(), desired_class_id.end(), class_id) == desired_class_id.end())
                continue;

            float cur_confidence = boxInfo.confidence;   // 置信度
            Rect cur_rect = boxInfo.box;                 // 目标框

            cur_rect &= Rect(Point(), frame.size());
            if (cur_rect.empty())
                continue;

            valid_objs.emplace_back(TrackedObject(cur_rect, cur_confidence, frame_idx, -1));
        }
        return valid_objs;
    }

    TrackedObjects TrackerByMatcher::track(const cv::Mat &frame, const std::vector<DetectBox> &origin_bboxes,
                                           unsigned long long frame_counter, TrackedObjects *detectedObject) {

        TrackedObjects valid_objs = this->nms(frame, origin_bboxes, frame_counter);
        if (detectedObject)
            *detectedObject = valid_objs;

        auto cur_timestamp = static_cast<uint64_t>(1000.0 / video_fps * frame_counter);
        tracker->process(frame, valid_objs, cur_timestamp);   // 跟踪核心函数


        TrackedObjects trackedObject = tracker->trackedDetections();

//        /// 记录每个跟踪框的第一次被跟踪的帧数
//        for (const TrackedObject &obj : trackedObject)
//            if (object_record.find(obj.object_id) == object_record.end()) {
//                object_record[obj.object_id] = obj.frame_idx;
//            }
        return trackedObject;
    }

}




