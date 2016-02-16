#include <opencv2/core/core.hpp>
#include <opencv/ml.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream> //for csv write
#include <vector>
#include <functional>
#include <ctime>
#include <random>
#include "ransac.h"


using namespace std;
using namespace cv;

typedef pair<float, float> dataType;
typedef pair<float, float> modelType; // y = mx + c ; m = first, c = second

void writePointstoFile(const vector<pair<float, float>>& points, const string file)
{
  string fname = "./" + file;
  ofstream fndAT(fname);
  for (auto& p : points)
  {
    fndAT << p.first << "," << p.second << endl;
  }
}

void openCVFitUsingLeastSquares(Mat& data)
{
  //void fitLine(InputArray points, OutputArray line, int distType, double param, double reps, double aeps)
  Vec4f lineModel;
  fitLine(data, lineModel, CV_DIST_L2, 0, 0.01, 0.01);
  cout << "line fit using least squares -> " << endl;
  cout << " Model in collinear vector and points " << lineModel << endl;
  //convert this line model to the slope intercept form : y = mx + c
  float m = lineModel[1] / lineModel[0];
  float c = (lineModel[2] * lineModel[1]) / lineModel[0] + lineModel[3];
  cout << " Model in Slope Intercept form " << endl;
  cout << " slope (m): " << m << endl;
  cout << " intercept (c): " << c << endl;
  cout << endl;
}

vector<dataType> loadPointsFromFile(const string file)
{
  vector<dataType> samples;
  string fname = "./" + file;
  CvMLData mlData;
  mlData.read_csv(fname.c_str());
  const CvMat* tmp = mlData.get_values();
  cv::Mat data(tmp, true);
  for (int i = 0; i < data.rows; i++)
  {
    float *e = data.ptr<float>(i);
    samples.push_back(dataType(e[0], e[1]));
  }
  // a sanity check on terminal display (uncomment for_each)
  //for_each(samples.begin(), samples.end(), [](dataType &d)
  //       {cout << d.first << " " << d.second << endl;});
  openCVFitUsingLeastSquares(data);
  return samples;
}


// this is about computing model parameters for a 2d line using ransac
int main(int argc, char ** argv)
{
  vector<dataType> dataPoints = loadPointsFromFile("foo.csv"); //example from scikit
  modelType groundTruthModel(0.2, 20); // example from scikit
  //parameters for ransac
  float distThresh = 0.05; //tolerance
  uint nbSamples = 2; // 2 points are needed to estimate a line
  float probInlierSet = 0.99; // How confident are we that we find a set containing inliers
  float proportionInliers = 0.7; // proportion of inliers in the dataset
  uint mxIt = 100; // max number of iterations for model fitting

  //functions for internal ransac functions to compute and assess model
  function<float(shared_ptr<dataType>& p, modelType& model)> distF =
    [](shared_ptr<dataType>& p, modelType& model)
    {
      //distance of a point from a line
      return fabs(p->second - model.first * p->first - model.second) /
      (sqrt(1 + pow(model.first,2) + pow(model.second, 2)));
    };

  function<modelType(vector<shared_ptr<dataType>>& data)> modelEst =
    [](vector<shared_ptr<dataType>>& data)
    {
      modelType m;
      //todo - degenerate case for line slope, divide by 0
      m.first = (data[1]->second - data[0]->second) / (data[1]->first - data[0]->first);
      m.second = data[0]->second - m.first*data[0]->first;
      return m;
    };

  // instantiate and call ransac
  modelType lineModel;
  ransac<dataType, modelType> lineFitUsingRansac(probInlierSet, mxIt, distThresh, modelEst,
                                                 distF, nbSamples, proportionInliers, dataPoints);
  lineModel = lineFitUsingRansac.fitModelToData();
  //display on terminal
  cout << "line fit using RANSAC -> " << endl;
  cout << " slope (m): " << lineModel.first << endl;
  cout << " intercept (c): " << lineModel.second << endl << endl;;
  cout << "** Gtruth line equation -> " << endl;
  cout << " slope (m): " << groundTruthModel.first << endl;
  cout << " intercept (c): " << groundTruthModel.second << endl;

  return 0;
}
