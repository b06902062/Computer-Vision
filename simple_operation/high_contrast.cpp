#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<iostream>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main() {
	Mat img = imread("lena.bmp", CV_8UC1);
	int rows = img.rows, cols = img.cols;
	int array[4][256] = { 0 };

	Mat hw1 = img.clone();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			int tmp = hw1.at<uchar>(i, j);
			array[0][tmp]++;
		}
	
	//for (int i = 0; i < 256; i++)		cout << array[0][i] << endl;
	//imshow("hw3-1", hw1);	imwrite("hw3-1.jpg", hw1);	waitKey(0);

	Mat hw2 = img.clone();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			hw2.at<uchar>(i, j) /= 3;
			int tmp = hw2.at<uchar>(i, j);
			array[1][tmp]++;
		}

	//for (int i = 0; i < 256; i++)		cout << array[1][i] << endl;
	//imshow("hw3-2", hw2);	imwrite("hw3-2.jpg", hw2);	waitKey(0);
	

	for (int i = 0; i < 256; i++) {
		if (i == 0)
			array[2][i] = array[1][i];
		else
			array[2][i] = array[1][i] + array[2][i-1];
	}
	
	Mat hw3 = hw2.clone();
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			int tmp = hw3.at<uchar>(i, j);
			hw3.at<uchar>(i, j) = array[2][tmp]*255/262144;
			tmp = hw3.at<uchar>(i, j);
			array[3][tmp]++;
		}

	//for (int i = 0; i < 256; i++)		cout << array[3][i] << endl;
	//imshow("hw3-3", hw3);	imwrite("hw3-3.jpg", hw3);	waitKey(0);

	return 0;
}