#include <opencv2/opencv.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

cv::Mat calnSkeleton(cv::Mat source) {
    cv::Mat image = source.clone();
    cv::Mat skeleton(image.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    
    bool done = false;
    while (!done) {
        cv::erode(image, eroded, element);
        cv::dilate(eroded, temp, element);
        cv::subtract(image, temp, temp);
        cv::bitwise_or(skeleton, temp, skeleton);
        eroded.copyTo(image);
        
        done = (cv::countNonZero(image) == 0);
    }
    
    return skeleton;
}

cv::Mat calnEdge(cv::Mat source) {
    cv::Mat edge(source.size(), CV_8UC1, cv::Scalar(0));
    cv::Canny(source, edge, 30, 100);
    return edge;
}

std::vector<cv::Point2f> getPixelSet(cv::Mat source) {
    std::vector<cv::Point2f> pixelSet;
    for (int r = 0; r < source.rows; r++) {
        for (int c = 0; c < source.cols; c++) {
            if (source.at<uchar>(r, c) > 127) {
                pixelSet.push_back(cv::Point2f(r, c));
            }
        }
    }
    return pixelSet;
}

float distanceBetweenPoints(cv::Point2f A, cv::Point2f B) {
    return sqrt((float)(A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
}

float distanceFromPointToPixelSet(cv::Point2f point, std::vector<cv::Point2f>& pixelSet, cv::flann::Index& kdtree, cv::Point2f& nearestPoint) {
    std::vector<float> query;
    query.push_back(point.x);
    query.push_back(point.y);
    std::vector<int> indices(1);
    std::vector<float> dists(1);
    cv::flann::SearchParams params(128);
    kdtree.knnSearch(query, indices, dists, 1, params);
    
    nearestPoint = pixelSet[indices[0]];
    return std::sqrt(dists[0]);
}

float distanceFromPointToPixelSet(cv::Point2f point, std::vector<cv::Point2f>& pixelSet, cv::flann::Index& kdtree) {
    cv::Point2f nearestPoint;
    return distanceFromPointToPixelSet(point, pixelSet, kdtree, nearestPoint);
}

void fitValueWithRank(std::vector<float> values, float& k, float& b) {
    std::vector<float> rank = values;
    std::sort(rank.begin(), rank.end());
    
    std::vector<cv::Point> points;
    for (int i = 0; i < rank.size(); i++) {
        points.push_back(cv::Point(i, rank[i]));
    }
    
    cv::Vec4f linePara;
    cv::fitLine(points, linePara, cv::DIST_L2, 0, 1e-2, 1e-2);
    
    k = linePara[1] / linePara[0];
    b = linePara[3] - k * linePara[2];
    
    std::cout << "Line regression of r(q) and rank(q): k = " << k << ", b = " << b << std::endl;
}

cv::Mat getNormalizedMap(cv::Mat source) {
    cv::Mat normalizedSource(source.size(), CV_8UC1);
    double minValue = 0;
    double maxValue = 0;
    cv::minMaxLoc(source, &minValue, &maxValue);
    for (int r = 0; r < source.rows; r++) {
        for (int c = 0; c < source.cols; c++) {
            normalizedSource.at<uchar>(r, c) = (source.at<float>(r, c) - minValue) / (maxValue - minValue) * 255;
        }
    }
    return normalizedSource;
}

cv::Mat getHeatMap(cv::Mat source)
{
    cv::Mat normalizedSource = getNormalizedMap(source);
    cv::Mat result;
    cv::applyColorMap(normalizedSource, result, cv::COLORMAP_HSV);
    return result;
}

cv::Mat normalizedDistanceEstimation(cv::Mat source) {
    cv::Mat sourceSkeleton = calnSkeleton(source);
    cv::Mat sourceEdge = calnEdge(source);
    
    std::vector<cv::Point2f> sourceSet = getPixelSet(source);
    std::vector<cv::Point2f> skeletonSet = getPixelSet(sourceSkeleton);
    std::vector<cv::Point2f> edgeSet = getPixelSet(sourceEdge);
    
    int nSource = (int)sourceSet.size();
    int nSkeleton = (int)skeletonSet.size();
    int nEdge = (int)edgeSet.size();
    std::cout << "|Sigma| = " << nSource << std::endl;
    std::cout << "|edge(Sigma)| = " << nSkeleton << std::endl;
    std::cout << "|skel(Sigma)| = " << nEdge << std::endl;
    
    cv::imshow("Distance Map", sourceSkeleton);
    cvWaitKey(0);
    
    cv::flann::KDTreeIndexParams indexParams(2);
    cv::flann::Index edgeKDTree(cv::Mat(edgeSet).reshape(1), indexParams);
    cv::flann::Index skelKDTree(cv::Mat(skeletonSet).reshape(1), indexParams);
    
    std::vector<float> rQ;
    for (int i = 0; i < nEdge; i++) {
        rQ.push_back(distanceFromPointToPixelSet(edgeSet[i], skeletonSet, skelKDTree));
    }
    float k = 0, b = 0;
    fitValueWithRank(rQ, k, b);
    
    float meanTextWidth = 0.5 * k * nEdge + b;
    float minRQ = 0.2 * k * nEdge + b;
    std::cout << "Mean Text Width = " << meanTextWidth << std::endl;
    
    
    cv::Mat result(source.size(), CV_32F, cv::Scalar(0));
    std::cout << "Distance Estimating: ";
    for (int r = 0; r < result.rows; r++) {
        if (r % (result.rows / 9) == 0 || r == result.rows - 1) {
            std::cout << ".";
        }
        for (int c = 0; c < result.cols; c++) {
            cv::Point2f q(r, c);
            
            cv::Point2f qNearest;
            float dist = distanceFromPointToPixelSet(q, edgeSet, edgeKDTree, qNearest);
            
            if (source.at<uchar>(r, c) > 127) {
                // inside Source
                dist = 1 - dist / std::max(minRQ, distanceFromPointToPixelSet(qNearest, skeletonSet, skelKDTree));
            } else {
                // outside Source
                dist = 1 + dist / meanTextWidth;
            }
            
            result.at<float>(r, c) = dist;
        }
    }
    std::cout << std::endl;
    
    return result;
}

void resizeImage(std::string fileName) {
    // Resize image to 512 * 512
    cv::Mat source = cv::imread(fileName, 1);
    cv::resize(source, source, cv::Size(512, 512));
    cv::imwrite("_" + fileName, source);
}

float distanceToCorrespondance(std::vector<float> query, cv::flann::Index& kdtree) {
    std::vector<int> indices(10);
    std::vector<float> dists(10);
    cv::flann::SearchParams params(128);
    kdtree.knnSearch(query, indices, dists, 10, params);
    
    return std::sqrt(dists[1]);
}

cv::Mat patchScaleDetection(cv::Mat source, cv::Mat source_) {
    // Define M = 3
    const int L = 5;
    const float W = 50;
    
    std::cout << "Patch Scale Detection Start" << std::endl;
    
    cv::cvtColor(source_, source_, cv::COLOR_RGB2GRAY);
    
    cv::Mat result(source.size(), CV_32F, cv::Scalar(0));
    
    for (int l = L; l >= 2; l--) {
        std::cout << l << " ";
        int s = 1 << (l - 1);
        cv::Mat scaled;
        cv::Mat scaled_;
        cv::resize(source, scaled, cv::Size(source.rows / s, source.cols / s));
        cv::resize(source_, scaled_, cv::Size(source_.rows / s, source_.cols / s));
        
        cv::Mat features((scaled.rows - 2) * (scaled.cols - 2), 9, CV_32FC1);
        cv::Mat features_((scaled_.rows - 2) * (scaled_.cols - 2), 9, CV_32FC1);
        
        int id = 0;
        for (int r = 1; r + 1 < scaled.rows; r++) {
            for (int c = 1; c + 1 < scaled.cols; c++) {
                if (result.at<float>(r * s, c * s) == 0) {
                    int k = 0;
                    for (int sr = r - 1; sr <= r + 1; sr++) {
                        for (int sc = c - 1; sc <= c + 1; sc++) {
                            features.at<float>(id, k) = (float)scaled.at<uchar>(sr, sc);
                            features_.at<float>(id, k) = (float)scaled_.at<uchar>(sr, sc);
                            k++;
                        }
                    }
                }
            }
            id++;
        }
        
        
        cv::flann::KDTreeIndexParams indexParams(9);
        cv::flann::Index kdtree(features, indexParams, cvflann::FLANN_DIST_EUCLIDEAN);
        cv::flann::Index kdtree_(features_, indexParams, cvflann::FLANN_DIST_EUCLIDEAN);
        
        for (int r = 1; r + 1 < scaled.rows; r++) {
            for (int c = 1; c + 1 < scaled.cols; c++) {
                if (result.at<float>(r * s, c * s) == 0) {
                    std::vector<float> feature;
                    std::vector<float> feature_;
                    for (int sr = r - 1; sr <= r + 1; sr++) {
                        for (int sc = c - 1; sc <= c + 1; sc++) {
                            feature.push_back((float)scaled.at<uchar>(sr, sc));
                            feature_.push_back((float)scaled_.at<uchar>(sr, sc));
                        }
                    }
                    float dist = distanceToCorrespondance(feature, kdtree);
                    float dist_ = distanceToCorrespondance(feature_, kdtree_);
                    
                    float dL = sqrt(dist * dist + dist_ * dist_);
                    
                    float sum = 0;
                    for (int i = 0; i < feature_.size(); i++) {
                        sum += feature_[i];
                    }
                    float mean = sum / feature_.size();
                    
                    float accum  = 0.0;
                    for (int i = 0; i < feature_.size(); i++) {
                        accum += (feature_[i] - mean) * (feature_[i] - mean);
                    }
                    float sigmaL = sqrt(accum/(feature_.size()-1)) / 2;
                    
                    if (dL + sigmaL < W) {
                        for (int sr = r * s; sr < (r + 1) * s; sr++) {
                            for (int sc = c * s; sc < (c + 1) * s; sc++) {
                                result.at<float>(sr, sc) = l;
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "1" << std::endl;
    std::cout << "Patch Scale Detection End" << std::endl;
    
    return result;
}

std::vector<cv::Point> getUnfilledPixels(cv::Mat valid) {
    std::vector<cv::Point> pixelByDegree[9];
    
    for (int r = 0; r < valid.rows; r++) {
        for (int c = 0; c < valid.cols; c++) {
            if (valid.at<float>(r, c) == 0) {
                int degree = 0;
                for (int sr = std::max(0, r - 1); sr < std::min(r + 2, valid.rows); sr++) {
                    for (int sc = std::max(0, c - 1); sc < std::min(c + 2, valid.cols); sc++) {
                        if (sr != r || sc != c) {
                            degree += valid.at<float>(sr, sc);
                        }
                    }
                }
                if (degree != 0) {
                    pixelByDegree[degree].push_back(cv::Point(r, c));
                }
            }
        }
    }
    
    std::vector<cv::Point> pixelList;
    
    for (int d = 8; d >= 1; d--) {
        std::vector<int> order(pixelByDegree[d].size());
        for (int i = 0; i < order.size(); i++) {
            order[i] = i;
        }
        for (int i = 0; i < order.size(); i++) {
            int j = rand() % (i + 1);
            std::swap(order[i], order[j]);
        }
        for (int i = 0; i < order.size(); i++) {
            pixelList.push_back(pixelByDegree[d][order[i]]);
        }
    }
    
    return pixelList;
}

cv::Mat gaussianKernel(cv::Size size, float sigma) {
    cv::Mat gauss2D(size, CV_32F, cv::Scalar(0));
    
    for (int r = 0; r < gauss2D.rows; r++) {
        for (int c = 0; c < gauss2D.cols; c++) {
            float dr = r - gauss2D.rows / 2;
            float dc = c - gauss2D.cols / 2;
            float dist2 = dr * dr + dc * dc;
            gauss2D.at<float>(r, c) = expf(-dist2 / (2 * sigma * sigma));
        }
    }
    
    return gauss2D;
}

cv::Point randomPickBestMatches(cv::Mat temp, cv::Mat valid, cv::Mat sample, float& error) {
    const float errThreshold = 0.1;
    const float sigma = sqrt(temp.rows * temp.cols) / 6.4;
    cv::Mat gauss2D = gaussianKernel(temp.size(), sigma);
    cv::Mat kernel(gauss2D.size(), CV_32F, cv::Scalar(0));
    
    float totWeight = 0;
    for (int r = 0; r < kernel.rows; r++) {
        for (int c = 0; c < kernel.cols; c++) {
            kernel.at<float>(r, c) = gauss2D.at<float>(r, c) * valid.at<float>(r, c);
            totWeight += kernel.at<float>(r, c);
        }
    }
    kernel = kernel / totWeight;
    
    //cv::Mat ssd;
    //cv::filter2D(sample, ssd, -1, kernel);
    //ssd = ssd(cv::Rect(0, 0, ssd.cols - kernel.cols + 1, ssd.rows - kernel.rows + 1));
    
    cv::Mat ssd(cv::Size(sample.rows - kernel.rows + 1, sample.cols - kernel.cols + 1), CV_32F, cv::Scalar(0));
#pragma omp parallel for
    for (int r = 0; r < ssd.rows; r++) {
        for (int c = 0; c < ssd.cols; c++) {
            for (int rr = 0; rr < kernel.rows; rr++) {
                for (int cc = 0; cc < kernel.cols; cc++) {
                    float dist = temp.at<uchar>(rr, cc) - sample.at<uchar>(r + rr, c + cc);
                    float dist2 = dist * dist;
                    ssd.at<float>(r, c) += dist2 * kernel.at<float>(rr, cc);
                }
            }
            
            if (ssd.at<float>(r, c) == 0) {
                ssd.at<float>(r, c) = 255;
            }
        }
    }
    
    
    cv::Point bestMatch;
    
    double minSSD, maxSSD;
    cv::minMaxLoc(ssd, &minSSD, &maxSSD);
    float threshold = minSSD * (1 + errThreshold);
    
    int cnt = 0;
    for (int r = 0; r < ssd.rows; r++) {
        for (int c = 0; c < ssd.cols; c++) {
            if (ssd.at<float>(r, c) < threshold) {
                cnt++;
                if (rand() % cnt == 0) {
                    error = ssd.at<float>(r, c);
                    bestMatch = cv::Point(r + kernel.rows / 2, c + kernel.cols / 2);
                }
            }
        }
    }
    
    return bestMatch;
}

cv::Mat textureSynthesis(cv::Mat sample, cv::Size windowSize) {
    cv::Mat sampleGray;
    cv::cvtColor(sample, sampleGray, CV_BGR2GRAY);
    cv::Mat result(cv::Size(sample.rows * 3, sample.cols * 3), CV_8UC3);
    cv::Mat resultCenter = result(cv::Rect(sample.cols, sample.rows, sample.cols, sample.rows));
    sample.copyTo(resultCenter);
    
    cv::Mat validMask(result.size(), CV_32F, cv::Scalar(0));
    for (int r = sample.rows; r < sample.rows * 2; r++) {
        for (int c = sample.cols; c < sample.cols * 2; c++) {
            validMask.at<float>(r, c) = 1;
        }
    }
    
    float maxErrThreshold = 0.3;
    
    int remain = validMask.rows * validMask.cols - cv::sum(validMask)[0];
    
    while (remain > 0) {
        std::cout << remain << std::endl;
        
        bool progress = false;
        std::vector<cv::Point> pixelList = getUnfilledPixels(validMask);
        
        for (int i = 0; i < pixelList.size(); i++) {
            cv::Point p = pixelList[i];
            
            cv::Mat temp(windowSize, CV_8UC3, cv::Scalar(0));
            cv::Mat tempValid(windowSize, CV_32F, cv::Scalar(0));
            for (int r = 0; r < temp.rows; r++) {
                for (int c = 0; c < temp.cols; c++) {
                    int tr = p.x - windowSize.height / 2 + r;
                    int tc = p.y - windowSize.width / 2 + c;
                    if (0 <= tr && tr < result.rows && 0 <= tc && tc < result.cols) {
                        temp.at<cv::Vec3b>(r, c) = result.at<cv::Vec3b>(tr, tc);
                        tempValid.at<float>(r, c) = validMask.at<float>(tr, tc);
                    }
                }
            }
            cv::Mat tempGray;
            cv::cvtColor(temp, tempGray, CV_BGR2GRAY);
            
            float error = 0;
            cv::Point bestMatch = randomPickBestMatches(tempGray, tempValid, sampleGray, error);
            error /= 255;
            if (error < maxErrThreshold) {
                result.at<cv::Vec3b>(p.x, p.y) = sample.at<cv::Vec3b>(bestMatch.x, bestMatch.y);
                remain--;
                validMask.at<float>(p.x, p.y) = 1;
                progress =  true;
            }
        }
        
        if (progress == false) {
            maxErrThreshold *= 1.1;
        }
    }
    
    return result;
}

int main() {
    // Show Distance Map
    /*cv::Mat source = cv::imread("S.png", 0);
    cv::Mat sourceBinary(source.size(), CV_8UC1, cv::Scalar(0));
    cv::threshold(source, sourceBinary, 127, 255, cv::THRESH_BINARY);
    
    cv::Mat distanceMap = normalizedDistanceEstimation(sourceBinary);
    cv::Mat heatMap = getHeatMap(distanceMap);
    
    cv::Mat sourceEdge = calnEdge(sourceBinary);
    cv::Mat coloredEdge;
    cv::cvtColor(sourceEdge, coloredEdge, cv::COLOR_GRAY2RGB);
    cv::Mat combination(source.size(), CV_8UC3, cv::Scalar(0));
    cv::bitwise_or(coloredEdge, heatMap, combination);
    
    cv::imshow("Distance Map", combination);
    cvWaitKey(0);*/
    
    // Show Scaled Map
    /*cv::Mat source = cv::imread("S.png", 0);
    cv::Mat source_ = cv::imread("S_.png", 1);
    cv::Mat patchScaleMap = patchScaleDetection(source, source_);
    cv::Mat normalizedMap = getNormalizedMap(patchScaleMap);
    cv::imshow("Scaled Map", normalizedMap);
    cvWaitKey(0);*/
    
    //Texture Synthesis
    cv::Mat sample = cv::imread("2.png", 1);
    cv::Mat result = textureSynthesis(sample, cv::Size(5, 5));
    cv::imshow("Texture Synthesis", result);
    cvWaitKey(0);
    cv::imwrite("result.jpg", result);
    
    return 0;
}
