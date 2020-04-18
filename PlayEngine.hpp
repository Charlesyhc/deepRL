#ifndef PLAY_ENGINE_HPP
#define PLAY_ENGINE_HPP
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>
#include <iostream>

#include "Param.hpp"

struct WFcolor
{
	uint8_t r;
	uint8_t g;
	uint8_t b;
};
void generateColorMap();
WFcolor bytesTovec3(uint8_t byte);




using Spectrum = std::array<uint8_t, SpecLen>;
using SpectrumSp = std::shared_ptr<Spectrum>;
//using Tone=std::array<uint8_t,50>;
struct Tone
{
	uint32_t startFreq;
	uint32_t bandWidth;
	std::vector<uint8_t> spec;
	uint32_t sweepSpeed;
	
};

class Jammer
{
protected:
	uint32_t _counter;
	Spectrum _spec;
	
	uint32_t _sweep_speed;
	uint32_t _band_width;
	
	std::vector<Tone> combSig;
	std::vector<Tone> sweepSig;
	std::vector<Tone>  pectinationSig;
	std::vector<Tone> smartSig;
	//std::vector<float> 
	
	float actionCounts[5];
	
	float curSmartFreq[3];
	
	int timePast;
	int timeTH;
	
	
public:
	Jammer();
	~Jammer(){};
	
	void Action();
	void Sweep();
	void Random();
	void Comb();
	void Smart();
	Spectrum *Spec(){return &_spec;}
	void resetCounts();
	void updateCounts(int action);
	void getSmartAction();
};


class PlayEngine
{
protected:
	uint32_t _ObserveTime;
	uint32_t _SpecLen;
	Spectrum _curSp;
	std::deque<Spectrum> _Waterfall;
	cv::Mat _waterFallImage;
	cv::Mat _bigWaterFallImage;
	cv::Mat _curSpecImage;
	Jammer jam;
	
	Tone userSig;
	int userPower;

	int _center_freq;
	int _user_band_width;
public:
	PlayEngine();
	~PlayEngine(){};
	
	void Show();
	float Step(int action); //获得效益
	std::deque<Spectrum> & Waterfall(){ return _Waterfall;};
	int getCurState();
};

void Sort(int a[], int n, int id[], int m);
int getMax(float data[], int num);
#endif








