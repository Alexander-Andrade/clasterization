#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <ctime>
#include <string.h>
#include <functional>
#include <algorithm>
#include <set>
#include <vector>
#include <list>
#include <memory>
#include <cmath>
#include <iterator>
#include <random>
#include <limits>


using namespace cv;
using namespace std;

enum Color { Black = 0, White = 255 };

class Timer
{
	clock_t t0;
	double cv_t0;
public:
	Timer() : t0(0) {}
	void start() { t0 = clock(); }
	void cv_start() { cv_t0 = (double)getTickCount(); }
	double difftime() { return ((double)(clock() - t0)) / (double)CLOCKS_PER_SEC; }
	double cv_difftime() { return ((double)getTickCount() - cv_t0) / getTickFrequency(); }
};

//number of attrs for 
static const int _nAttrs = 4;

class ObjAttrs {
	//claster of this obj
	int _clasterNum;
	int _id;
	Rect _bound;

	Mat _labels;

	int _area;
	Point2d _massCenter;
	int _perimeter;
	double _density;
	double _elongation;
	double _inertAxisOrient;

	double m_20;
	double m_02;
	double m_11;

	Vec<double, _nAttrs> _featureVec;

	void getBounds() {
		Size size = _labels.size();
		int* row;
		Point a;
		Point b;
		bool flinside = false;
		for (int y = 0; y < size.height; y++) {
			row = _labels.ptr<int>(y);
			for (int x = 0; x < size.width; x++) {
				if (row[x] == _id && (!flinside)) {
					a = Point(x, y);
					b = a;
					flinside = true;
				}
				else if (row[x] == _id && flinside) {
					a.x = std::min(a.x, x);

					b.x = std::max(b.x, x);
					b.y = std::max(b.y, y);
				}
			}
		}
		b += Point(1, 1);
		_bound = Rect(a, b);
	}

	bool isInnerBoundary(int x, int y) {
		//four connectivity
		int* cur_line = _labels.ptr<int>(y);
		if (x > 0 && cur_line[ x - 1] != _id)
			return true;
		if ((x < _labels.size().height - 1) && cur_line[x + 1] != _id)
			return true;
		if (y > 0 && _labels.ptr<int>(y - 1)[x] != _id)
			return true;
		if (y < _labels.size().width && _labels.ptr<int>(y + 1)[x] != _id)
			return true;
		return false;
	}

	void calcAreaPerimMassCenter() {
		int y_end = _bound.y + _bound.height;
		int x_end = _bound.x + _bound.width;
		int* row;
		for (int y = _bound.y; y < y_end; y++) {
			row = _labels.ptr<int>(y);
			for (int x = _bound.x; x < x_end; x++) {
				if (row[x] == _id) {
					//inc area
					_area++;
					//inc perimeter
					if (isInnerBoundary(x, y))
						_perimeter++;
					//mass center
					_massCenter.x += x;
					_massCenter.y += y;
				}
			}
		}
		_massCenter.x /= _area;
		_massCenter.y /= _area;
	}

	void calcDencity() {
		_density = pow(_perimeter, 2) / _area;
	}

	double discreteCentralMoment(int i, int j) {
		int y_end = _bound.y + _bound.height;
		int x_end = _bound.x + _bound.width;
		int* row;
		double moment = 0;
		for (int y = _bound.y; y < y_end; y++) {
			row = _labels.ptr<int>(y);
			for (int x = _bound.x; x < x_end; x++) {
				if (row[x] == _id) {
					moment += pow(x - _massCenter.x, i) * pow(y - _massCenter.y, j);
				}
			}
		}
		return moment;
	}

	void calcCentralMoments() {
		m_20 = discreteCentralMoment(2, 0);
		m_02 = discreteCentralMoment(0, 2);
		m_11 = discreteCentralMoment(1, 1);
	}

	void calcElongation() {
		if (_area == 1) {
			_elongation = 0;
			return;
		}

		double m_20_plus_m_02 = m_20 + m_02;
		double square_m_20_minus_m_02 = pow(m_20 - m_02,2);
		double squares_sum_root = sqrt(square_m_20_minus_m_02 + 4 * pow(m_11, 2));
		_elongation = (m_20_plus_m_02 + squares_sum_root) / (m_20_plus_m_02 - squares_sum_root);
	}

	void calcInertAxisOrient() {
		if (_area == 1) {
			_inertAxisOrient = 0;
			return;
		}
		_inertAxisOrient = 0.5 * atan(2 * m_11 / (m_20 - m_02));
	}

public:
	ObjAttrs(Mat& labels, int id) : _id(id), _clasterNum(0), _labels(labels), _bound(Rect(Point(0, 0), Point(0, 0))), _area(0), _massCenter(Point2d(0., 0.)),
		_perimeter(0), _density(0), _inertAxisOrient(0), _elongation(0), m_02(0.), m_20(0.), m_11(0.)
	{
		//find bound rect
		getBounds();
	}

	void calc() {
		calcAreaPerimMassCenter();
		calcDencity();

		calcCentralMoments();
		calcElongation();
		calcInertAxisOrient();
		_featureVec = { (double)_area,(double)_perimeter,_density,_elongation };
	}

	double& operator[](int i) { return _featureVec[i]; }

	Vec<double, _nAttrs> getFeatureVec() { return _featureVec; }

	int id()const { return _id; }
	Rect bound()const { return _bound; }
	int area()const { return _area; }
	Point2d massCenter()const { return _massCenter; }
	int perimeter()const { return _perimeter; }
	double density()const { return _density; }
	double inertAxisOrient()const { return _inertAxisOrient; }
	double elongatuion()const { return _elongation; }
	int getClasterNum()const { return _clasterNum; }
	void setClasterNum(int clasterNum) { _clasterNum = clasterNum; }

	bool isObjPoint(int x, int y) { return _labels.at<int>(y, x) == _id; }
	bool intersects(ObjAttrs& enother) {
		Rect intersectRect = _bound & enother.bound();

		if (intersectRect.width == 0 && intersectRect.height == 0)
			return false;

		int y_end = intersectRect.y + intersectRect.height;
		int x_end = intersectRect.x + intersectRect.width;

		for (int y = intersectRect.y; y < y_end; y++)
			for (int x = intersectRect.x; x < x_end; x++)
				if (isObjPoint(x, y) && enother.isObjPoint(x, y)) {
					return true;
				}
		return false;
	}

	friend ostream& operator<<(ostream& stream, ObjAttrs& obj) {
		stream << "id: " << obj._id << endl <<
			"claster: " << obj._clasterNum << endl <<
			"area: " << obj._area << endl <<
			"mass center: x " << obj._massCenter.x << "y " << obj._massCenter.y << endl <<
			"perimetr: " << obj._perimeter << endl <<
			"dencity: " << obj._density << endl <<
			"m_20: " << obj.m_20 << "  m_02: " << obj.m_02 << "  m_11: " << obj.m_11 << endl <<
			"inert axis orient: " << obj._inertAxisOrient << endl <<
			"elongation: " << obj._elongation << endl;
		stream << "attrs: ";
		for (int i = 0; i < _nAttrs; i++)
			stream << "x" + std::to_string(i) << ": " << obj._featureVec[i] << ", ";
		return stream;
	}
};

class K_means {
	std::string _srcName;
	int _nClasters;
	vector<Vec<double, _nAttrs>> _meanClasterVects;
	vector < Vec<double, _nAttrs>> _prevMeanClasterVects;

	std::default_random_engine _generator;
	std::uniform_real_distribution<double>  _real_dist;
	std::uniform_int_distribution<int>  _int_dist;

	double real_rand() { return _real_dist(_generator); }
	int int_rand() { return _int_dist(_generator); }

	static void toGrayscale(Mat& src, Mat& dst, double r = 0.3, double g = 0.59, double b = 0.11) {
		//8 bit matrix
		CV_Assert(src.depth() == CV_8U);
		CV_Assert(src.channels() == 3);

		dst = Mat::zeros(src.size(), src.type());
		MatIterator_<Vec3b> itSrc, endSrc, itDst, endDst;

		for (itSrc = src.begin<Vec3b>(), itDst = dst.begin<Vec3b>(), endDst = dst.end<Vec3b>(), endSrc = src.end<Vec3b>(); itSrc != endSrc; ++itSrc, ++itDst) {
			uchar y = r * (*itSrc)[0] + g * (*itSrc)[1] + b* (*itSrc)[2];
			(*itDst)[0] = (*itDst)[1] = (*itDst)[2] = y;
		}
	}

	static void showFilter(Mat& src) {
		Mat nIm;

		Mat kernel = (Mat_<double>(3, 3) << 0, 1, 0,
			1, 0, -1,
			0, -1, 0);
		/*
		ddepth = depth of dst A negative value (such as -1) indicates that the depth is the same as the source.
		anchor: The position of the anchor relative to its kernel. The location Point(-1, -1) indicates the center by default.
		delta: A value to be added to each pixel during the convolution. By default it is 0
		*/
		Timer t;
		t.cv_start();
		filter2D(src, nIm, -1, kernel);
		cout << t.cv_difftime() << endl;
		imshow("qwerty", nIm);
		waitKey(0);
	}

	void imTraverse(Mat& binIm, Mat& labels, int x, int y, int domainNum) {
		//0 - black color
		//255 - white
		//cout << "x: " << x << "y: " << y << "domain: " << domainNum << endl;
		int* label_row = labels.ptr<int>(y);
		if (label_row[x] == 0 && binIm.ptr<uchar>(y)[x] == White) {
			label_row[x] = domainNum;

			if (x > 0)
				imTraverse(binIm, labels, x - 1, y, domainNum);
			if (x < binIm.size().width - 1)
				imTraverse(binIm, labels, x + 1, y, domainNum);
			if (y > 0)
				imTraverse(binIm, labels, x, y - 1, domainNum);
			if (y < binIm.size().height - 1)
				imTraverse(binIm, labels, x, y + 1, domainNum);
		}
	}

	void labeling(Mat& binIm, Mat& labels) {
		Size size = binIm.size();
		int domainNum = 1;

		for (int y = 0; y < size.height; y++)
			for (int x = 0; x < size.width; x++)
				imTraverse(binIm, labels, x, y, domainNum++);

	}

	std::set<int> listIds(Mat& im) {
		Size s = im.size();
		set<int> ids;
		std::set<int>::iterator it;
		int prevId = 0;
		int* p;
		for (int y = 0; y < s.height; y++) {
			p = im.ptr<int>(y);
			for (int x = 0; x < s.width; x++) {
				if (p[x] != 0 && p[x] != prevId) {
					it = ids.find(p[x]);
					if (it == ids.end())
						ids.insert(p[x]);
					prevId = p[x];
				}
			}
		}
		return ids;
	}

	void createObjects(Mat& labels, vector<ObjAttrs>& objects) {
		//fill objects and calc attributes
		std::set<int> ids = listIds(labels);
		objects.reserve(ids.size());
		for (auto id : ids)
			objects.push_back(ObjAttrs(labels, id));
	}
	void calcAttrs(std::vector<ObjAttrs>& objects) {
		//calculate attrs
		for (ObjAttrs& obj : objects)
			obj.calc();
	}
	void filterMessByArea(std::vector<ObjAttrs>& objects,int area) {
		//filter small objects area < 20
		objects.erase(std::remove_if(objects.begin(), objects.end(), [area](const ObjAttrs& obj) {return obj.area() < area; }), objects.end());
	}

	auto minMaxOfAttr(vector<ObjAttrs>& objects, int i) {
		auto compare = [i](ObjAttrs& a, ObjAttrs& b) {return a[i] < b[i]; };
		return std::minmax_element(objects.begin(), objects.end(), compare);
	}

	void initMeanVectorsFromObjects(vector<ObjAttrs>& objects) {
		_meanClasterVects.reserve(_nClasters);
		//set init mean vectors from existing first vectors
		for (int i = 0; i < _nClasters; i++)
			_meanClasterVects.push_back(objects[i].getFeatureVec());
		//empty previous clasters centers vector
		_prevMeanClasterVects.resize(_nClasters);
	}

	Vec<double, _nAttrs> randFeatureVec() {
		Vec<double, _nAttrs> v;
		for (int i = 0; i < _nAttrs; i++) 
			v[i] = real_rand();
		return v;
	}

	void initMeanVectorsFromObjMeanAndRand(vector<ObjAttrs>& objects) {
		//!!! only apply after normalization
		_meanClasterVects.resize(_nClasters);
		
		for (int clN = 0; clN < _nClasters; clN++)
			_meanClasterVects[clN] = randFeatureVec();
		//empty previous clasters centers vector
		_prevMeanClasterVects.resize(_nClasters);
	}

	double euclideanDistance(Vec<double, _nAttrs>& a, Vec<double, _nAttrs> b) {
		auto res = a - b;
		return norm(res);
	}

	void normalizeAttrs(vector<ObjAttrs>& objects) {
		double min = 0.0;
		double max = 0.0;
		for (int i = 0; i < _nAttrs; i++) {
			auto it = minMaxOfAttr(objects,i);
			min = (*it.first)[i];
			max = (*it.second)[i];
			std::for_each(objects.begin(), objects.end(), [i, min, max](ObjAttrs& obj) {
				obj[i] = (obj[i] - min) / (max - min);
			});
		}
	}
	int countClasterObjects(vector<ObjAttrs>& objects,int clasterN) {
		return std::count_if(objects.begin(), objects.end(), [clasterN](ObjAttrs& obj) {return obj.getClasterNum() == clasterN; });
	}
	void classifyObj(ObjAttrs& obj) {
		double* clasters_dist = new double[_nClasters];
		for (int i = 0; i < _nClasters; i++)
			clasters_dist[i] = euclideanDistance(_meanClasterVects[i], obj.getFeatureVec());

		int clasterNum = std::distance(clasters_dist, std::min_element(clasters_dist, clasters_dist + _nClasters));
		obj.setClasterNum(clasterNum);
		delete[] clasters_dist;
	}

	void classifyAllObjs(vector<ObjAttrs>& objects) {
		std::for_each(objects.begin(), objects.end(), [this](ObjAttrs& obj) {classifyObj(obj); });
	}

	double averageForClasterAttr(vector<ObjAttrs>& objects,int clasterNum, int attrNum) {
		double min = 0.0;
		double max = 0.0;
		bool flFirst = false;
		for (ObjAttrs& obj : objects) {
			if (obj.getClasterNum() == clasterNum) {
				if (flFirst) {
					min = std::min(min, obj[attrNum]);
					max = std::max(max, obj[attrNum]);
				}
				else {
					min = obj[attrNum];
					max = obj[attrNum];
					flFirst = true;
				}

			}
		}
		return (min + max) / 2;
	}
	ostream& showMeanVect(int clasterN, ostream& stream = cout) {
		stream << "MEAN VECTOR" + to_string(clasterN) << ": " << _meanClasterVects[clasterN] << endl;
		return stream;
	}
	void showMeanVects() {
		for (int i = 0; i < _meanClasterVects.size(); i++)
			showMeanVect(i);
	}
	void recalcClastersMassCenter(vector<ObjAttrs>& objects) {
		for (int clNum = 0; clNum < _nClasters; clNum++) {
			if (countClasterObjects(objects, clNum) == 0)
				_meanClasterVects[clNum] = randFeatureVec();
			else
				for (int attrNum = 0; attrNum < _nAttrs; attrNum++)
					_meanClasterVects[clNum][attrNum] = averageForClasterAttr(objects, clNum, attrNum);
			//showMeanVects();
		}
	}

	void divideObjectsToClasters(vector<ObjAttrs>& objects) {
		normalizeAttrs(objects);
		//initMeanVectorsFromObjects(objects);
		initMeanVectorsFromObjMeanAndRand(objects);
		while (_meanClasterVects != _prevMeanClasterVects) {
			classifyAllObjs(objects);
			_prevMeanClasterVects = _meanClasterVects;
			recalcClastersMassCenter(objects);
		}

	}
	void logObjAttrs(vector<ObjAttrs>& objects) {
		//log objects attrs
		for (int clasterN = 0; clasterN < _nClasters; clasterN++) {
			cout << endl << endl << endl;
			showMeanVect(clasterN) << endl;
			for (ObjAttrs& obj : objects)
				if (obj.getClasterNum() == clasterN)
					cout << obj << endl << endl;
		}
	}
	void printClastersColors(std::vector<Scalar>& clastColors) {
		int height = 20;
		int width = 240;
		int nColors = clastColors.size();
		Size size(width, height * nColors);
		Rect colorRect(Point(0,0),Point(width,height));
		Mat colorIm = Mat::zeros(size, CV_8UC3);
		for (int i = 0; i < nColors; i++) {
			rectangle(colorIm,colorRect, clastColors[i],-1);
			colorRect.y += height;
		}
		imshow("clasters colors",colorIm);
	}
	int randIntensity() { return int_rand() % 255; }
	void printClastersObjectsBoundRects(Mat& img,vector<ObjAttrs>& objects) {
		Scalar color;
		std::vector<Scalar> colors;
		colors.reserve(_nClasters);
		//print rectangles
		for (int clNum = 0; clNum < _nClasters; clNum++) {
			color[0] = randIntensity();
			color[1] = randIntensity();
			color[2] = randIntensity();
			colors.push_back(color);
			std::for_each(objects.begin(), objects.end(), [clNum, &color, &img, this](ObjAttrs& obj) {
				if (obj.getClasterNum() == clNum)
					rectangle(img, obj.bound(), color);
			});
		}
		//colors table
		printClastersColors(colors);
	}
	void delDisappearedObjects(vector<ObjAttrs>& objects1, vector<ObjAttrs>& objects2) {
		objects1.erase(std::remove_if(objects1.begin(), objects1.end(), [&objects2](ObjAttrs& obj1) {
			return !std::any_of(objects2.begin(), objects2.end(), [&obj1](ObjAttrs& obj2) {return obj2.intersects(obj1); });
		}),objects1.end());
	}
	
public:
	K_means(const char* srcName, int nClasters) : _srcName(srcName), _nClasters(nClasters) {
		_generator.seed(time(NULL));
	}
	void run() {
		// e_xxxxx  ->enchanced 
		//load image
		Mat src = imread(_srcName);
		//imshow("original", src);
		//convert it to grayscale
		Mat gray = Mat::zeros(src.size(), CV_8UC1);
		cvtColor(src, gray, COLOR_RGB2GRAY);
		//make grayscale copy and apply filters
		Mat e_gray = Mat::zeros(gray.size(), gray.type());
		GaussianBlur(gray, e_gray, Size(25,25),0);
		medianBlur(e_gray, e_gray, 7);
		//GaussianBlur(gray, e_gray, Size(3, 3), 0);
		//medianBlur(gray, filtered, 7);
		//adaptiveBilateralFilter(gray, e_gray, Size(5,5), 100, 100);
		//blur(gray, e_gray, Size(5, 5));
		//edge detection
		Mat sob_x_gray, sob_y_gray;
		Mat abs_soby_gray, abs_sobx_gray;
		Mat sob_res;
		Sobel(gray, sob_x_gray, CV_16S, 1, 0);
		convertScaleAbs(sob_x_gray, abs_sobx_gray);
		Sobel(gray, sob_y_gray, CV_16S, 0, 1);
		convertScaleAbs(sob_y_gray, abs_soby_gray);
		addWeighted(abs_sobx_gray, 0.5, abs_soby_gray, 0.5, 0, sob_res);
		Mat s_binarized;
		threshold(sob_res, s_binarized, 0, White, THRESH_BINARY | THRESH_OTSU);
		imshow("bin",s_binarized);
		
		//binarize normal and filtered grayscale images
		Mat binarized, e_binarized;
		threshold(gray, binarized, 0, White, THRESH_BINARY | THRESH_OTSU);
		threshold(e_gray, e_binarized, 0, White, THRESH_BINARY | THRESH_OTSU);
		//threshold(b_gray, b_binarized, 0, White, THRESH_BINARY | THRESH_OTSU);
		Mat edged_binarized = binarized & (~s_binarized);
		imshow("binarized", binarized);
		imshow("filtered binarized", e_binarized);
		imshow("test", edged_binarized);
		
		//create two matrixes for labeling (objects identification)
		Mat labels = Mat::zeros(binarized.size(), CV_32SC1);
		Mat e_labels = Mat::zeros(e_binarized.size(), CV_32SC1);
		labeling(binarized,labels);
		labeling(e_binarized, e_labels);
		
		//objects creation
		vector<ObjAttrs> objects;
		vector<ObjAttrs> e_objects;
		createObjects(labels,objects);
		createObjects(e_labels,e_objects);
		//cut off those objects which are not present in the second image
		delDisappearedObjects(objects, e_objects);
		//calculate objects attributes values
		calcAttrs(objects);
		//clasterization
		divideObjectsToClasters(objects);
		//select objects belonging to the same class with one color
		printClastersObjectsBoundRects(src,objects);
		imshow("original with selected objects", src);
		//show attributes of the objects
		logObjAttrs(objects);
		
		waitKey(0);
	}
};

int main()
{
	srand(time(NULL));
	/*
	#pragma omp parallel for
	for (int i = 0; i < size; i++)
	*/
	K_means k_means("P0001469.jpg", 4);
	k_means.run();
	system("pause");
    return 0;

}