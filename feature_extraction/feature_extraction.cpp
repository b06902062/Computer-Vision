#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdlib.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <math.h> 

using namespace cv;
using namespace std;

Mat extend(Mat img, int k) {
	Mat tmp(img.rows + 2*k, img.cols + 2*k, CV_8UC1);
	for (int i = 0; i < tmp.rows; i++)
		for (int j = 0; j < tmp.cols; j++) {
			if (i < k) {
				if (j < k)	tmp.at<uchar>(i, j) = img.at<uchar>(0, 0);
				else if (j > tmp.cols-1-k) tmp.at<uchar>(i, j) = img.at<uchar>(0, img.cols-1);
				else tmp.at<uchar>(i, j) = img.at<uchar>(0, j - k);
			}
			else if (i > tmp.rows-1-k) {
				if (j < k)	tmp.at<uchar>(i, j) = img.at<uchar>(img.rows - 1, 0);
				else if (j > tmp.cols-1-k) tmp.at<uchar>(i, j) = img.at<uchar>(img.rows - 1, img.cols - 1);
				else tmp.at<uchar>(i, j) = img.at<uchar>(img.rows - 1, j - k);
			}
			else if (j < k)	tmp.at<uchar>(i, j) = img.at<uchar>(i - k, 0);
			else if (j > tmp.cols-1-k)	tmp.at<uchar>(i, j) = img.at<uchar>(i - k, img.cols-1);
			else	tmp.at<uchar>(i, j) = img.at<uchar>(i - k, j - k);
		}
	return tmp;
}

Mat laplace1(Mat img, int threshold) {
	int test;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			test = img.at<uchar>(i - 1, j) + img.at<uchar>(i + 1, j) + img.at<uchar>(i, j - 1)
				+ img.at<uchar>(i, j + 1) - 4 * img.at<uchar>(i, j);
			tmp.at<uchar>(i - 1, j - 1) = test > threshold ? 0 : 255;
		}
	return tmp;
}

Mat laplace2(Mat img, int threshold) {
	double test;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			test = img.at<uchar>(i - 1, j - 1) + img.at<uchar>(i - 1, j) + img.at<uchar>(i - 1, j + 1)
				+ img.at<uchar>(i, j - 1) +(-8)*img.at<uchar>(i, j) + img.at<uchar>(i, j + 1)
				+ img.at<uchar>(i + 1, j - 1) + img.at<uchar>(i + 1, j) + img.at<uchar>(i + 1, j + 1);
			test /= 3;
			tmp.at<uchar>(i - 1, j - 1) = test > threshold ? 0 : 255;
		}
	return tmp;
}

Mat laplacem(Mat img, int threshold) {
	double test;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			test = 2 * img.at<uchar>(i - 1, j - 1) - img.at<uchar>(i - 1, j) + 2 * img.at<uchar>(i - 1, j + 1)
					 - img.at<uchar>(i, j - 1) - 4 * img.at<uchar>(i, j) - img.at<uchar>(i, j + 1)
				 + 2 * img.at<uchar>(i + 1, j - 1) - img.at<uchar>(i + 1, j) + 2 * img.at<uchar>(i + 1, j + 1);
			test /= 3;
			tmp.at<uchar>(i - 1, j - 1) = test > threshold ? 0 : 255;
		}
	return tmp;
}

Mat laplaceG(Mat img, int threshold) {
	long long int test;
	Mat tmp(img.rows - 10, img.cols - 10, CV_8UC1);
	for (int i = 5; i < img.rows - 5; i++)
		for (int j = 5; j < img.cols - 5; j++) {
			test = -img.at<uchar>(i - 5, j - 2) - img.at<uchar>(i - 5, j - 1) - 2 * img.at<uchar>(i - 5, j) - img.at<uchar>(i - 5, j + 1) - img.at<uchar>(i - 5, j + 2)
				- 2 * img.at<uchar>(i - 4, j - 3) - 4 * img.at<uchar>(i - 4, j - 2) - 8 * img.at<uchar>(i - 4, j - 1) - 9 * img.at<uchar>(i - 4, j)
				- 2 * img.at<uchar>(i - 4, j + 3) - 4 * img.at<uchar>(i - 4, j + 2) - 8 * img.at<uchar>(i - 4, j + 1)
				- 2 * img.at<uchar>(i - 3, j - 4) - 7 * img.at<uchar>(i - 3, j - 3) - 15 * img.at<uchar>(i - 3, j - 2) - 22 * img.at<uchar>(i - 3, j - 1) - 23 * img.at<uchar>(i - 3, j)
				- 2 * img.at<uchar>(i - 3, j + 4) - 7 * img.at<uchar>(i - 3, j + 3) - 15 * img.at<uchar>(i - 3, j + 2) - 22 * img.at<uchar>(i - 3, j + 1)
				- img.at<uchar>(i - 2, j - 5) - 4 * img.at<uchar>(i - 2, j - 4) - 15 * img.at<uchar>(i - 2, j - 3) - 24 * img.at<uchar>(i - 2, j - 2) - 14 * img.at<uchar>(i - 2, j - 1) - img.at<uchar>(i - 2, j)
				- img.at<uchar>(i - 2, j + 5) - 4 * img.at<uchar>(i - 2, j + 4) - 15 * img.at<uchar>(i - 2, j + 3) - 24 * img.at<uchar>(i - 2, j + 2) - 14 * img.at<uchar>(i - 2, j + 1)
				- img.at<uchar>(i - 1, j - 5) - 8 * img.at<uchar>(i - 1, j - 4) - 22 * img.at<uchar>(i - 1, j - 3) - 14 * img.at<uchar>(i - 1, j - 2) + 52 * img.at<uchar>(i - 1, j - 1) + 103 * img.at<uchar>(i - 1, j)
				- img.at<uchar>(i - 1, j + 5) - 8 * img.at<uchar>(i - 1, j + 4) - 22 * img.at<uchar>(i - 1, j + 3) - 14 * img.at<uchar>(i - 1, j + 2) + 52 * img.at<uchar>(i - 1, j + 1)
				- 2 * img.at<uchar>(i, j - 5) - 9 * img.at<uchar>(i, j - 4) - 23 * img.at<uchar>(i, j - 3) - img.at<uchar>(i, j - 2) + 103 * img.at<uchar>(i, j - 1) + 178 * img.at<uchar>(i, j)
				- 2 * img.at<uchar>(i, j + 5) - 9 * img.at<uchar>(i, j + 4) - 23 * img.at<uchar>(i, j + 3) - img.at<uchar>(i, j + 2) + 103 * img.at<uchar>(i, j + 1)
				- img.at<uchar>(i + 1, j - 5) - 8 * img.at<uchar>(i + 1, j - 4) - 22 * img.at<uchar>(i + 1, j - 3) - 14 * img.at<uchar>(i + 1, j - 2) + 52 * img.at<uchar>(i + 1, j - 1) + 103 * img.at<uchar>(i + 1, j)
				- img.at<uchar>(i + 1, j + 5) - 8 * img.at<uchar>(i + 1, j + 4) - 22 * img.at<uchar>(i + 1, j + 3) - 14 * img.at<uchar>(i + 1, j + 2) + 52 * img.at<uchar>(i + 1, j + 1)
				- img.at<uchar>(i + 2, j - 5) - 4 * img.at<uchar>(i + 2, j - 4) - 15 * img.at<uchar>(i + 2, j - 3) - 24 * img.at<uchar>(i + 2, j - 2) - 14 * img.at<uchar>(i + 2, j - 1) - img.at<uchar>(i + 2, j)
				- img.at<uchar>(i + 2, j + 5) - 4 * img.at<uchar>(i + 2, j + 4) - 15 * img.at<uchar>(i + 2, j + 3) - 24 * img.at<uchar>(i + 2, j + 2) - 14 * img.at<uchar>(i + 2, j + 1)
				- 2 * img.at<uchar>(i + 3, j - 4) - 7 * img.at<uchar>(i + 3, j - 3) - 15 * img.at<uchar>(i + 3, j - 2) - 22 * img.at<uchar>(i + 3, j - 1) - 23 * img.at<uchar>(i + 3, j)
				- 2 * img.at<uchar>(i + 3, j + 4) - 7 * img.at<uchar>(i + 3, j + 3) - 15 * img.at<uchar>(i + 3, j + 2) - 22 * img.at<uchar>(i + 3, j + 1)
				- 2 * img.at<uchar>(i + 4, j - 3) - 4 * img.at<uchar>(i + 4, j - 2) - 8 * img.at<uchar>(i + 4, j - 1) - 9 * img.at<uchar>(i + 4, j)
				- 2 * img.at<uchar>(i + 4, j + 3) - 4 * img.at<uchar>(i + 4, j + 2) - 8 * img.at<uchar>(i + 4, j + 1)
				- img.at<uchar>(i + 5, j - 2) - img.at<uchar>(i + 5, j - 1) - 2 * img.at<uchar>(i + 5, j) - img.at<uchar>(i + 5, j + 1) - img.at<uchar>(i + 5, j + 2);
			tmp.at<uchar>(i - 5, j - 5) = test > threshold ? 0 : 255;
		}
	return tmp;
}

Mat DiffGaus(Mat img, int threshold) {
	long long int test;
	Mat tmp(img.rows - 10, img.cols - 10, CV_8UC1);
	for (int i = 5; i < img.rows - 5; i++)
		for (int j = 5; j < img.cols - 5; j++) {
			test = -img.at<uchar>(i - 5, j - 5) - 3 * img.at<uchar>(i - 5, j - 4) - 4 * img.at<uchar>(i - 5, j - 3) - 6 * img.at<uchar>(i - 5, j - 2) - 7 * img.at<uchar>(i - 5, j - 1) - 8 * img.at<uchar>(i - 5, j)
				- img.at<uchar>(i - 5, j + 5) - 3 * img.at<uchar>(i - 5, j + 4) - 4 * img.at<uchar>(i - 5, j + 3) - 6 * img.at<uchar>(i - 5, j + 2) - 7 * img.at<uchar>(i - 5, j + 1)
				- 3 * img.at<uchar>(i - 4, j - 5) - 5 * img.at<uchar>(i - 4, j - 4) - 8 * img.at<uchar>(i - 4, j - 3) - 11 * img.at<uchar>(i - 4, j - 2) - 13 * img.at<uchar>(i - 4, j - 1) - 13 * img.at<uchar>(i - 4, j)
				- 3 * img.at<uchar>(i - 4, j + 5) - 5 * img.at<uchar>(i - 4, j + 4) - 8 * img.at<uchar>(i - 4, j + 3) - 11 * img.at<uchar>(i - 4, j + 2) - 13 * img.at<uchar>(i - 4, j + 1)
				- 4 * img.at<uchar>(i - 3, j - 5) - 8 * img.at<uchar>(i - 3, j - 4) - 12 * img.at<uchar>(i - 3, j - 3) - 16 * img.at<uchar>(i - 3, j - 2) - 17 * img.at<uchar>(i - 3, j - 1) - 17 * img.at<uchar>(i - 3, j)
				- 4 * img.at<uchar>(i - 3, j + 5) - 8 * img.at<uchar>(i - 3, j + 4) - 12 * img.at<uchar>(i - 3, j + 3) - 16 * img.at<uchar>(i - 3, j + 2) - 17 * img.at<uchar>(i - 3, j + 1)
				- 6 * img.at<uchar>(i - 2, j - 5) - 11 * img.at<uchar>(i - 2, j - 4) - 16 * img.at<uchar>(i - 2, j - 3) - 16 * img.at<uchar>(i - 2, j - 2) + 15 * img.at<uchar>(i - 2, j)
				- 6 * img.at<uchar>(i - 2, j + 5) - 11 * img.at<uchar>(i - 2, j + 4) - 16 * img.at<uchar>(i - 2, j + 3) - 16 * img.at<uchar>(i - 2, j + 2)
				- 7 * img.at<uchar>(i - 1, j - 5) - 13 * img.at<uchar>(i - 1, j - 4) - 17 * img.at<uchar>(i - 1, j - 3) + 85 * img.at<uchar>(i - 1, j - 1) + 160 * img.at<uchar>(i - 1, j)
				- 7 * img.at<uchar>(i - 1, j + 5) - 13 * img.at<uchar>(i - 1, j + 4) - 17 * img.at<uchar>(i - 1, j + 3) + 85 * img.at<uchar>(i - 1, j + 1)
				- 8 * img.at<uchar>(i, j - 5) - 13 * img.at<uchar>(i, j - 4) - 17 * img.at<uchar>(i, j - 3) + 15 * img.at<uchar>(i, j - 2) + 160 * img.at<uchar>(i, j - 1) + 283 * img.at<uchar>(i, j)
				- 8 * img.at<uchar>(i, j + 5) - 13 * img.at<uchar>(i, j + 4) - 17 * img.at<uchar>(i, j + 3) + 15 * img.at<uchar>(i, j + 2) + 160 * img.at<uchar>(i, j + 1)
				- 7 * img.at<uchar>(i + 1, j - 5) - 13 * img.at<uchar>(i + 1, j - 4) - 17 * img.at<uchar>(i + 1, j - 3) + 85 * img.at<uchar>(i + 1, j - 1) + 160 * img.at<uchar>(i + 1, j)
				- 7 * img.at<uchar>(i + 1, j + 5) - 13 * img.at<uchar>(i + 1, j + 4) - 17 * img.at<uchar>(i + 1, j + 3) + 85 * img.at<uchar>(i + 1, j + 1)
				- 6 * img.at<uchar>(i + 2, j - 5) - 11 * img.at<uchar>(i + 2, j - 4) - 16 * img.at<uchar>(i + 2, j - 3) - 16 * img.at<uchar>(i + 2, j - 2) + 15 * img.at<uchar>(i + 2, j)
				- 6 * img.at<uchar>(i + 2, j + 5) - 11 * img.at<uchar>(i + 2, j + 4) - 16 * img.at<uchar>(i + 2, j + 3) - 16 * img.at<uchar>(i + 2, j + 2)
				- 4 * img.at<uchar>(i + 3, j - 5) - 8 * img.at<uchar>(i + 3, j - 4) - 12 * img.at<uchar>(i + 3, j - 3) - 16 * img.at<uchar>(i + 3, j - 2) - 17 * img.at<uchar>(i + 3, j - 1) - 17 * img.at<uchar>(i + 3, j)
				- 4 * img.at<uchar>(i + 3, j + 5) - 8 * img.at<uchar>(i + 3, j + 4) - 12 * img.at<uchar>(i + 3, j + 3) - 16 * img.at<uchar>(i + 3, j + 2) - 17 * img.at<uchar>(i + 3, j + 1)
				- 3 * img.at<uchar>(i + 4, j - 5) - 5 * img.at<uchar>(i + 4, j - 4) - 8 * img.at<uchar>(i + 4, j - 3) - 11 * img.at<uchar>(i + 4, j - 2) - 13 * img.at<uchar>(i + 4, j - 1) - 13 * img.at<uchar>(i + 4, j)
				- 3 * img.at<uchar>(i + 4, j + 5) - 5 * img.at<uchar>(i + 4, j + 4) - 8 * img.at<uchar>(i + 4, j + 3) - 11 * img.at<uchar>(i + 4, j + 2) - 13 * img.at<uchar>(i + 4, j + 1)
				- img.at<uchar>(i + 5, j - 5) - 3 * img.at<uchar>(i + 5, j - 4) - 4 * img.at<uchar>(i + 5, j - 3) - 6 * img.at<uchar>(i + 5, j - 2) - 7 * img.at<uchar>(i + 5, j - 1) - 8 * img.at<uchar>(i + 5, j)
				- img.at<uchar>(i + 5, j + 5) - 3 * img.at<uchar>(i + 5, j + 4) - 4 * img.at<uchar>(i + 5, j + 3) - 6 * img.at<uchar>(i + 5, j + 2) - 7 * img.at<uchar>(i + 5, j + 1);
			tmp.at<uchar>(i - 5, j - 5) = test > threshold ? 255:0;
		}
	return tmp;
}

int main() {
	Mat img = imread("lena.bmp", CV_8UC1);

	Mat lap1 = laplace1(extend(img, 1), 18);	imshow("lap1", lap1);	imwrite("lap1.jpg", lap1);	waitKey(0);
	Mat lap2 = laplace2(extend(img, 1), 15);	imshow("lap2", lap2);	imwrite("lap2.jpg", lap2);	waitKey(0);
	Mat lapm = laplacem(extend(img, 1), 11);	imshow("lapm", lapm);	imwrite("lapm.jpg", lapm);	waitKey(0);
	Mat lapG = laplaceG(extend(img, 5), 3000);	imshow("lapG", lapG);	imwrite("lapG.jpg", lapG);	waitKey(0);
	Mat DffG = DiffGaus(extend(img, 5), 41000);	imshow("DffG", DffG);	imwrite("DffG.jpg", DffG);	waitKey(0);

	return 0;
}