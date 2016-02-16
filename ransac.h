#ifndef _RANSAC_H
#define _RANSAC_H

#include <iostream>
#include <math.h>
#include <exception>
#include <numeric>
#include <vector>
#include <algorithm>
#include <memory>
using namespace std;


template <typename DataType, typename Model>
class ransac
{
  typedef shared_ptr<DataType> ptrDataType;
 private:
  uint _nbIterations;
  uint _maxIterations;
  uint _nbSamples;
  float _tolerance;
  vector<ptrDataType> _allSamples;
  function<Model(vector<ptrDataType>& )>& _estimator; //todo compare time with cref and no-& here
  function<float(ptrDataType& , Model& )>& _distFunction;

 public:
  ransac(float pInlierSet, uint maxIt, float tolerance, function<Model(vector<ptrDataType>& samples)>&
         est, function<float(ptrDataType& sample, Model& model)>& dist,
         uint nbSamples, float propInliers, vector<DataType>& data) :
  _estimator(est),  _distFunction(dist),
    _tolerance (tolerance), _nbSamples(nbSamples), _maxIterations(maxIt)
  {
    _allSamples.resize(data.size());
    transform(data.begin(), data.end(), _allSamples.begin(), [](DataType& d)
              {return make_shared<DataType>(d);});
    //compute _nbIter
    _nbIterations = static_cast<int>(ceil(log(1 - pInlierSet) /
                                          (log(1- pow(propInliers, _nbSamples)))));
  }

  vector<float> computeModelFitErrors(Model& tentativeModel, vector<ptrDataType>& data)
  {
    vector<float> errors(data.size(), 0.0);
    transform(data.begin(), data.end(), errors.begin(), [this, &tentativeModel](ptrDataType ptD)
              {return this->_distFunction(ptD, tentativeModel);});
    return errors;
  }

  vector<ptrDataType> chooseRandomSubset()
  {
    random_shuffle(_allSamples.begin(), _allSamples.end());
    vector<ptrDataType> subset;
    copy(_allSamples.begin(), _allSamples.begin() + _nbSamples, back_inserter(subset));
    return subset;
  }

  Model fitModelToData()
  {
    cout << "Start Ransac: " << endl;
    cout << "Required Number of iterations (theoretically) = " << _nbIterations << endl;
    uint iterations = 0;
    Model outputModel;
    uint outputInlierCount = 0;
    while (iterations < _maxIterations && iterations < _nbIterations)
    {
      vector<ptrDataType> dataSubset = chooseRandomSubset();
      Model tentativeModel = _estimator(dataSubset);
      vector<float> errors = computeModelFitErrors(tentativeModel, _allSamples);
      uint nbInliers = count_if(errors.begin(), errors.end(), [this](float t)
      {return t < _tolerance;});
      if (nbInliers > outputInlierCount)
      {
        outputInlierCount = nbInliers;
        outputModel = tentativeModel;
      }
      iterations++;
    }
    return outputModel;
    //todo - degenerate model check
  }
};

#endif //
