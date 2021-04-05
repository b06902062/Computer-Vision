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

Mat Robert(Mat img, int threshold) {
	int r1, r2;	double gradient;
	Mat tmp(img.rows-2, img.cols-2, CV_8UC1);
	for (int i = 1; i < img.rows-1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			r1 = img.at<uchar>(i + 1, j + 1) - img.at<uchar>(i, j);
			r2 = img.at<uchar>(i + 1, j) - img.at<uchar>(i, j + 1);
			gradient = sqrt(r1*r1+r2*r2);
			tmp.at<uchar>(i - 1, j - 1) = gradient > threshold ? 0 : 255;
		}
	return tmp;
}

Mat Prewitt(Mat img, int threshold) {
	int p1, p2;	double gradient;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			p1 = img.at<uchar>(i + 1, j - 1) + img.at<uchar>(i + 1, j) + img.at<uchar>(i + 1, j + 1)
				- img.at<uchar>(i - 1, j - 1) - img.at<uchar>(i - 1, j) - img.at<uchar>(i - 1, j + 1);
			p2 = img.at<uchar>(i - 1, j + 1) + img.at<uchar>(i, j + 1) + img.at<uchar>(i + 1, j + 1)
				- img.at<uchar>(i - 1, j - 1) - img.at<uchar>(i, j - 1) - img.at<uchar>(i + 1, j - 1);
			gradient = sqrt(p1 * p1 + p2 * p2);
			tmp.at<uchar>(i - 1, j - 1) = gradient > threshold ? 0 : 255;
		}
	return tmp;
}

Mat Sobel(Mat img, int threshold) {
	int s1, s2;	double gradient;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			s1 = img.at<uchar>(i + 1, j - 1) + 2 * img.at<uchar>(i + 1, j) + img.at<uchar>(i + 1, j + 1)
				- img.at<uchar>(i - 1, j - 1) - 2 * img.at<uchar>(i - 1, j) - img.at<uchar>(i - 1, j + 1);
			s2 = img.at<uchar>(i - 1, j + 1) + 2 * img.at<uchar>(i, j + 1) + img.at<uchar>(i + 1, j + 1)
				- img.at<uchar>(i - 1, j - 1) - 2 * img.at<uchar>(i, j - 1) - img.at<uchar>(i + 1, j - 1);
			gradient = sqrt(s1 * s1 + s2 * s2);
			tmp.at<uchar>(i - 1, j - 1) = gradient > threshold ? 0 : 255;
		}
	return tmp;
}

Mat Frei_Chen(Mat img, int threshold) {
	double f1, f2, gradient;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			f1 = img.at<uchar>(i + 1, j - 1) + sqrt(2) * img.at<uchar>(i + 1, j) + img.at<uchar>(i + 1, j + 1)
				- img.at<uchar>(i - 1, j - 1) - sqrt(2) * img.at<uchar>(i - 1, j) - img.at<uchar>(i - 1, j + 1);
			f2 = img.at<uchar>(i - 1, j + 1) + sqrt(2) * img.at<uchar>(i, j + 1) + img.at<uchar>(i + 1, j + 1)
				- img.at<uchar>(i - 1, j - 1) - sqrt(2) * img.at<uchar>(i, j - 1) - img.at<uchar>(i + 1, j - 1);
			gradient = sqrt(f1 * f1 + f2 * f2);
			tmp.at<uchar>(i - 1, j - 1) = gradient > threshold ? 0 : 255;
		}
	return tmp;
}

Mat Kirsch(Mat img, int threshold) {
	int k[8];	double gradient;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			k[0] = -3 * img.at<uchar>(i - 1, j - 1) - 3 * img.at<uchar>(i - 1, j) + 5 * img.at<uchar>(i - 1, j + 1)
				- 3 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) + 5 * img.at<uchar>(i, j + 1)
				- 3 * img.at<uchar>(i + 1, j - 1) - 3 * img.at<uchar>(i + 1, j) + 5 * img.at<uchar>(i + 1, j + 1);
			k[1] = -3 * img.at<uchar>(i - 1, j - 1) + 5 * img.at<uchar>(i - 1, j) + 5 * img.at<uchar>(i - 1, j + 1)
				- 3 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) + 5 * img.at<uchar>(i, j + 1)
				- 3 * img.at<uchar>(i + 1, j - 1) - 3 * img.at<uchar>(i + 1, j) - 3 * img.at<uchar>(i + 1, j + 1);
			k[2] = 5 * img.at<uchar>(i - 1, j - 1) + 5 * img.at<uchar>(i - 1, j) + 5 * img.at<uchar>(i - 1, j + 1)
				- 3 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 3 * img.at<uchar>(i, j + 1)
				- 3 * img.at<uchar>(i + 1, j - 1) - 3 * img.at<uchar>(i + 1, j) - 3 * img.at<uchar>(i + 1, j + 1);
			k[3] = 5 * img.at<uchar>(i - 1, j - 1) + 5 * img.at<uchar>(i - 1, j) - 3 * img.at<uchar>(i - 1, j + 1)
				+ 5 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 3 * img.at<uchar>(i, j + 1)
				- 3 * img.at<uchar>(i + 1, j - 1) - 3 * img.at<uchar>(i + 1, j) - 3 * img.at<uchar>(i + 1, j + 1);
			k[4] = 5 * img.at<uchar>(i - 1, j - 1) - 3 * img.at<uchar>(i - 1, j) - 3 * img.at<uchar>(i - 1, j + 1)
				+ 5 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 3 * img.at<uchar>(i, j + 1)
				+ 5 * img.at<uchar>(i + 1, j - 1) - 3 * img.at<uchar>(i + 1, j) - 3 * img.at<uchar>(i + 1, j + 1);
			k[5] = -3 * img.at<uchar>(i - 1, j - 1) - 3 * img.at<uchar>(i - 1, j) - 3 * img.at<uchar>(i - 1, j + 1)
				+ 5 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 3 * img.at<uchar>(i, j + 1)
				+ 5 * img.at<uchar>(i + 1, j - 1) + 5 * img.at<uchar>(i + 1, j) - 3 * img.at<uchar>(i + 1, j + 1);
			k[6] = -3 * img.at<uchar>(i - 1, j - 1) - 3 * img.at<uchar>(i - 1, j) - 3 * img.at<uchar>(i - 1, j + 1)
				- 3 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 3 * img.at<uchar>(i, j + 1)
				+ 5 * img.at<uchar>(i + 1, j - 1) + 5 * img.at<uchar>(i + 1, j) + 5 * img.at<uchar>(i + 1, j + 1);
			k[7] = -3 * img.at<uchar>(i - 1, j - 1) - 3 * img.at<uchar>(i - 1, j) - 3 * img.at<uchar>(i - 1, j + 1)
				- 3 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) + 5 * img.at<uchar>(i, j + 1)
				- 3 * img.at<uchar>(i + 1, j - 1) + 5 * img.at<uchar>(i + 1, j) + 5 * img.at<uchar>(i + 1, j + 1);
			gradient = -1;
			for (int p = 0; p < 8; p++)
				if (k[p] > gradient) gradient = k[p];
			tmp.at<uchar>(i - 1, j - 1) = gradient > threshold ? 0 : 255;
		}
	return tmp;
}

Mat Robinson(Mat img, int threshold) {
	int r[8];	double gradient;
	Mat tmp(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++)
		for (int j = 1; j < img.cols - 1; j++) {
			r[0] = -1 * img.at<uchar>(i - 1, j - 1) + 0 * img.at<uchar>(i - 1, j) + 1 * img.at<uchar>(i - 1, j + 1)
				- 2 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) + 2 * img.at<uchar>(i, j + 1)
				- 1 * img.at<uchar>(i + 1, j - 1) + 0 * img.at<uchar>(i + 1, j) + 1 * img.at<uchar>(i + 1, j + 1);
			r[1] = 0 * img.at<uchar>(i - 1, j - 1) + 1 * img.at<uchar>(i - 1, j) + 2 * img.at<uchar>(i - 1, j + 1)
				- 1 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) + 1 * img.at<uchar>(i, j + 1)
				- 2 * img.at<uchar>(i + 1, j - 1) - 1 * img.at<uchar>(i + 1, j) + 0 * img.at<uchar>(i + 1, j + 1);
			r[2] = 1 * img.at<uchar>(i - 1, j - 1) + 2 * img.at<uchar>(i - 1, j) + 1 * img.at<uchar>(i - 1, j + 1)
				+ 0 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) + 0 * img.at<uchar>(i, j + 1)
				- 1 * img.at<uchar>(i + 1, j - 1) - 2 * img.at<uchar>(i + 1, j) - 1 * img.at<uchar>(i + 1, j + 1);
			r[3] = 2 * img.at<uchar>(i - 1, j - 1) + 1 * img.at<uchar>(i - 1, j) + 0 * img.at<uchar>(i - 1, j + 1)
				+ 1 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 1 * img.at<uchar>(i, j + 1)
				+ 0 * img.at<uchar>(i + 1, j - 1) - 1 * img.at<uchar>(i + 1, j) - 2 * img.at<uchar>(i + 1, j + 1);
			r[4] = 1 * img.at<uchar>(i - 1, j - 1) + 0 * img.at<uchar>(i - 1, j) - 1 * img.at<uchar>(i - 1, j + 1)
				+ 2 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 2 * img.at<uchar>(i, j + 1)
				+ 1 * img.at<uchar>(i + 1, j - 1) + 0 * img.at<uchar>(i + 1, j) - 1 * img.at<uchar>(i + 1, j + 1);
			r[5] = 0 * img.at<uchar>(i - 1, j - 1) - 1 * img.at<uchar>(i - 1, j) - 2 * img.at<uchar>(i - 1, j + 1)
				+ 1 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 1 * img.at<uchar>(i, j + 1)
				+ 2 * img.at<uchar>(i + 1, j - 1) + 1 * img.at<uchar>(i + 1, j) + 0 * img.at<uchar>(i + 1, j + 1);
			r[6] = -1 * img.at<uchar>(i - 1, j - 1) - 2 * img.at<uchar>(i - 1, j) - 1 * img.at<uchar>(i - 1, j + 1)
				+ 0 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) - 0 * img.at<uchar>(i, j + 1)
				+ 1 * img.at<uchar>(i + 1, j - 1) + 2 * img.at<uchar>(i + 1, j) + 1 * img.at<uchar>(i + 1, j + 1);
			r[7] = -2 * img.at<uchar>(i - 1, j - 1) - 1 * img.at<uchar>(i - 1, j) + 0 * img.at<uchar>(i - 1, j + 1)
				- 1 * img.at<uchar>(i, j - 1) + 0 * img.at<uchar>(i, j) + 1 * img.at<uchar>(i, j + 1)
				+ 0 * img.at<uchar>(i + 1, j - 1) + 1 * img.at<uchar>(i + 1, j) + 2 * img.at<uchar>(i + 1, j + 1);
			gradient = -1;
			for (int p = 0; p < 8; p++)
				if (r[p] > gradient) gradient = r[p];
			tmp.at<uchar>(i - 1, j - 1) = gradient > threshold ? 0 : 255;
		}
	return tmp;
}

Mat Nevatia_Babu(Mat img, int threshold) {
	int N[6];	double gradient;
	Mat tmp(img.rows - 4, img.cols - 4, CV_8UC1);
	for (int i = 2; i < img.rows - 2; i++)
		for (int j = 2; j < img.cols - 2; j++) {
			N[0] = 100 * img.at<uchar>(i - 2, j - 2) + 100 * img.at<uchar>(i - 2, j - 1) + 100 * img.at<uchar>(i - 2, j) + 100 * img.at<uchar>(i - 2, j + 1) + 100 * img.at<uchar>(i - 2, j + 2)
				+ 100 * img.at<uchar>(i - 1, j - 2) + 100 * img.at<uchar>(i - 1, j - 1) + 100 * img.at<uchar>(i - 1, j) + 100 * img.at<uchar>(i - 1, j + 1) + 100 * img.at<uchar>(i - 1, j + 2)
				- 100 * img.at<uchar>(i + 1, j - 2) - 100 * img.at<uchar>(i + 1, j - 1) - 100 * img.at<uchar>(i + 1, j) - 100 * img.at<uchar>(i + 1, j + 1) - 100 * img.at<uchar>(i + 1, j + 2)
				- 100 * img.at<uchar>(i + 2, j - 2) - 100 * img.at<uchar>(i + 2, j - 1) - 100 * img.at<uchar>(i + 2, j) - 100 * img.at<uchar>(i + 2, j + 1) - 100 * img.at<uchar>(i + 2, j + 2);

			N[1] = 100 * img.at<uchar>(i - 2, j - 2) + 100 * img.at<uchar>(i - 2, j - 1) + 100 * img.at<uchar>(i - 2, j) + 100 * img.at<uchar>(i - 2, j + 1) + 100 * img.at<uchar>(i - 2, j + 2)
				+ 100 * img.at<uchar>(i - 1, j - 2) + 100 * img.at<uchar>(i - 1, j - 1) + 100 * img.at<uchar>(i - 1, j) + 78 * img.at<uchar>(i - 1, j + 1) - 32 * img.at<uchar>(i - 1, j + 2)
				+ 100 * img.at<uchar>(i - 0, j - 2) + 92 * img.at<uchar>(i - 0, j - 1) + 0 * img.at<uchar>(i - 0, j) - 92 * img.at<uchar>(i - 0, j + 1) - 100 * img.at<uchar>(i - 0, j + 2)
				+ 32 * img.at<uchar>(i + 1, j - 2) - 78 * img.at<uchar>(i + 1, j - 1) - 100 * img.at<uchar>(i + 1, j) - 100 * img.at<uchar>(i + 1, j + 1) - 100 * img.at<uchar>(i + 1, j + 2)
				- 100 * img.at<uchar>(i + 2, j - 2) - 100 * img.at<uchar>(i + 2, j - 1) - 100 * img.at<uchar>(i + 2, j) - 100 * img.at<uchar>(i + 2, j + 1) - 100 * img.at<uchar>(i + 2, j + 2);

			N[2] = 100 * img.at<uchar>(i - 2, j - 2) + 100 * img.at<uchar>(i - 2, j - 1) + 100 * img.at<uchar>(i - 2, j) + 32 * img.at<uchar>(i - 2, j + 1) - 100 * img.at<uchar>(i - 2, j + 2)
				+ 100 * img.at<uchar>(i - 1, j - 2) + 100 * img.at<uchar>(i - 1, j - 1) + 92 * img.at<uchar>(i - 1, j) - 78 * img.at<uchar>(i - 1, j + 1) - 100 * img.at<uchar>(i - 1, j + 2)
				+ 100 * img.at<uchar>(i - 0, j - 2) + 100 * img.at<uchar>(i - 0, j - 1) + 0 * img.at<uchar>(i - 0, j) - 100 * img.at<uchar>(i - 0, j + 1) - 100 * img.at<uchar>(i - 0, j + 2)
				+ 100 * img.at<uchar>(i + 1, j - 2) + 78 * img.at<uchar>(i + 1, j - 1) - 92 * img.at<uchar>(i + 1, j) - 100 * img.at<uchar>(i + 1, j + 1) - 100 * img.at<uchar>(i + 1, j + 2)
				+ 100 * img.at<uchar>(i + 2, j - 2) - 32 * img.at<uchar>(i + 2, j - 1) - 100 * img.at<uchar>(i + 2, j) - 100 * img.at<uchar>(i + 2, j + 1) - 100 * img.at<uchar>(i + 2, j + 2);

			N[3] = -100 * img.at<uchar>(i - 2, j - 2) - 100 * img.at<uchar>(i - 2, j - 1) + 100 * img.at<uchar>(i - 2, j + 1) + 100 * img.at<uchar>(i - 2, j + 2)
				- 100 * img.at<uchar>(i - 1, j - 2) - 100 * img.at<uchar>(i - 1, j - 1) + 100 * img.at<uchar>(i - 1, j + 1) + 100 * img.at<uchar>(i - 1, j + 2)
				- 100 * img.at<uchar>(i - 0, j - 2) - 100 * img.at<uchar>(i - 0, j - 1) + 100 * img.at<uchar>(i - 0, j + 1) + 100 * img.at<uchar>(i - 0, j + 2)
				- 100 * img.at<uchar>(i + 1, j - 2) - 100 * img.at<uchar>(i + 1, j - 1) + 100 * img.at<uchar>(i + 1, j + 1) + 100 * img.at<uchar>(i + 1, j + 2)
				- 100 * img.at<uchar>(i + 2, j - 2) - 100 * img.at<uchar>(i + 2, j - 1) + 100 * img.at<uchar>(i + 2, j + 1) + 100 * img.at<uchar>(i + 2, j + 2);

			N[4] = -100 * img.at<uchar>(i - 2, j - 2) + 32 * img.at<uchar>(i - 2, j - 1) + 100 * img.at<uchar>(i - 2, j) + 100 * img.at<uchar>(i - 2, j + 1) + 100 * img.at<uchar>(i - 2, j + 2)
				- 100 * img.at<uchar>(i - 1, j - 2) - 78 * img.at<uchar>(i - 1, j - 1) + 92 * img.at<uchar>(i - 1, j) + 100 * img.at<uchar>(i - 1, j + 1) + 100 * img.at<uchar>(i - 1, j + 2)
				- 100 * img.at<uchar>(i - 0, j - 2) - 100 * img.at<uchar>(i - 0, j - 1) + 0 * img.at<uchar>(i - 0, j) + 100 * img.at<uchar>(i - 0, j + 1) + 100 * img.at<uchar>(i - 0, j + 2)
				- 100 * img.at<uchar>(i + 1, j - 2) - 100 * img.at<uchar>(i + 1, j - 1) - 92 * img.at<uchar>(i + 1, j) + 78 * img.at<uchar>(i + 1, j + 1) + 100 * img.at<uchar>(i + 1, j + 2)
				- 100 * img.at<uchar>(i + 2, j - 2) - 100 * img.at<uchar>(i + 2, j - 1) - 100 * img.at<uchar>(i + 2, j) - 32 * img.at<uchar>(i + 2, j + 1) + 100 * img.at<uchar>(i + 2, j + 2);

			N[5] = 100 * img.at<uchar>(i - 2, j - 2) + 100 * img.at<uchar>(i - 2, j - 1) + 100 * img.at<uchar>(i - 2, j) + 100 * img.at<uchar>(i - 2, j + 1) + 100 * img.at<uchar>(i - 2, j + 2)
				- 32 * img.at<uchar>(i - 1, j - 2) + 78 * img.at<uchar>(i - 1, j - 1) + 100 * img.at<uchar>(i - 1, j) + 100 * img.at<uchar>(i - 1, j + 1) + 100 * img.at<uchar>(i - 1, j + 2)
				- 100 * img.at<uchar>(i - 0, j - 2) - 92 * img.at<uchar>(i - 0, j - 1) + 0 * img.at<uchar>(i - 0, j) + 92 * img.at<uchar>(i - 0, j + 1) + 100 * img.at<uchar>(i - 0, j + 2)
				- 100 * img.at<uchar>(i + 1, j - 2) - 100 * img.at<uchar>(i + 1, j - 1) - 100 * img.at<uchar>(i + 1, j) - 78 * img.at<uchar>(i + 1, j + 1) + 32 * img.at<uchar>(i + 1, j + 2)
				- 100 * img.at<uchar>(i + 2, j - 2) - 100 * img.at<uchar>(i + 2, j - 1) - 100 * img.at<uchar>(i + 2, j) - 100 * img.at<uchar>(i + 2, j + 1) - 100 * img.at<uchar>(i + 2, j + 2);

			gradient = -1;
			for (int p = 0; p < 6; p++)
				if (N[p] > gradient) gradient = N[p];
			tmp.at<uchar>(i - 2, j - 2) = gradient > threshold ? 0 : 255;
		}

	return tmp;
}

int main() {
	Mat img = imread("lena.bmp", CV_8UC1);

	Mat rober = Robert(extend(img, 1), 15);	imshow("rober", rober);	imwrite("rober.jpg", rober);	waitKey(0);
	Mat prewi = Prewitt(extend(img, 1), 35);	imshow("prewi", prewi);	imwrite("prewi.jpg", prewi);	waitKey(0);
	Mat sobel = Sobel(extend(img, 1), 45);		imshow("sobel", sobel);	imwrite("sobel.jpg", sobel);	waitKey(0);
	Mat freic = Frei_Chen(extend(img, 1), 35);	imshow("freic", freic);	imwrite("freic.jpg", freic);	waitKey(0);
	Mat kirsc = Kirsch(extend(img, 1), 145);	imshow("kirsc", kirsc);	imwrite("kirsc.jpg", kirsc);	waitKey(0);
	Mat robin = Robinson(extend(img, 1), 50);	imshow("robin", robin);	imwrite("robin.jpg", robin);	waitKey(0);
	Mat nevat = Nevatia_Babu(extend(img, 1), 12500);	imshow("nevat", nevat);	imwrite("nevat.jpg", nevat);	waitKey(0);

	return 0;
}