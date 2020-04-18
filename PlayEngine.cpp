#include "PlayEngine.hpp"
#include <iostream>
#include <math.h>
#define pi 3.14159265
#define FLOOR_DBM -90
using namespace std;

WFcolor ColorMap[256];
void generateColorMap()
{
	const WFcolor red={255,0,0};
	const WFcolor yellow={255,255,0};
	const WFcolor green={0,255,0};
	const WFcolor cyan={0,255,255};
	const WFcolor blue={0,0,255};
	const WFcolor cp[5]={blue,cyan,green,yellow,red}; 
	
	uint8_t r,g,b;
	WFcolor start,stop;
	int steps;
	int range=64;
	for(auto byte=0;byte<256;byte++)
	{
		int pos=byte/range+1;

		start=cp[pos-1];
		stop=cp[pos];
		steps=byte-(pos-1)*range;
		ColorMap[byte].r=start.r+(stop.r-start.r)*steps/range;
		ColorMap[byte].g=start.g+(stop.g-start.g)*steps/range;
		ColorMap[byte].b=start.b+(stop.b-start.b)*steps/range;
	}
	
	
}




WFcolor bytesTovec3(uint8_t byte)
{
	return ColorMap[byte];
}

int logadd(int a, int b)
{
	float temp=pow(10,a/10)+pow(10,b/10);
	return 10*log10(temp);
}

float rcosine(int i, int B, float alpha)
{
	
	
	float band=B;
	float fi=fabs(i-band/2.0)*(1+alpha)/2; //只取有信号的地方
	float B1=(1-alpha)*band/4.0;
	float B2=(1+alpha)*band/4.0;
	std::cout<<fi<<" "<<B1<<" "<<B2<<std::endl;
	if(fi<=B1) return 1;
	if(fi<=B2) return 0.5* ( 1+cos((fi-B1)/band*3.14159265*2));
	return 0;
	
	
}

Jammer::Jammer()
{
	_counter=0;
	_band_width=40;
	_sweep_speed=10;

	
//构建扫频信号
	for(int i=0;i<1;i++)
	{
		Tone sig;
		sig.bandWidth=_band_width;
		sig.startFreq=_band_width*2*i+_band_width/2;
		sig.sweepSpeed=_sweep_speed;
		for(int ss=0;ss<sig.bandWidth;ss++)
		{
			uint8_t value=rcosine(ss,sig.bandWidth,0.5)*(30-FLOOR_DBM)*2;
			sig.spec.push_back(value);
		}
		sweepSig.push_back(sig);		
	}
	
//构建梳状干扰
	for(int i=0;i<3;i++)
	{
		Tone sig;
		sig.bandWidth=_band_width;
		sig.startFreq=_band_width*2*i+_band_width/2;
		sig.sweepSpeed=0;
		for(int ss=0;ss<sig.bandWidth;ss++)
		{
			uint8_t value=rcosine(ss,sig.bandWidth,0.5)*(30-FLOOR_DBM)*2-20*log10(3);
			sig.spec.push_back(value);
		}
		combSig.push_back(sig);		
	}
	
	
	resetCounts();	
	int SmartToneNum=3;
	for(int i=0;i<SmartToneNum;i++)
	{
		Tone sig;
		
		curSmartFreq[i]=_band_width*i+_band_width/2;
		
		sig.bandWidth=_band_width;
		sig.startFreq=_band_width*2*i+_band_width/2;
		sig.sweepSpeed=0;
		for(int ss=0;ss<sig.bandWidth;ss++)
		{
			uint8_t value=rcosine(ss,sig.bandWidth,0.5)*(30-FLOOR_DBM)*2-20*log10(SmartToneNum);
			sig.spec.push_back(value);
		}
		smartSig.push_back(sig);	
	}


	for(int i=0;i<5;i++)
	{
		actionCounts[i]=0;
	}
	
	
	
	

}
int d_counter=0;
void Jammer::Action()
{
	//Random();
	d_counter++;
	//int dd=d_counter/50;
//	if(dd%2==1)
//	{ 
//		Sweep();
//	}
//	else {
	//	Comb();
//	}
	//Sweep();
	//Comb();
	Sweep();
	//Smart();
}


void Jammer::resetCounts()
{
	//for(int i=0;i<5;i++) actionCounts[i]=0;
	//timePast=0;
}

void Jammer::updateCounts(int action)
{
	for(int i=0;i<5;i++)
	{
		actionCounts[i]=actionCounts[i]-0.3;
		if(actionCounts[i]<0) actionCounts[i]=0;
	}
	
	
	if(action % 2 ==1)
	{
		int jamAction=floor(action/2);
		actionCounts[jamAction]=actionCounts[jamAction]+1;	
	}
	else
	{
		int jamAction1=action/2;
		int jamAction2=jamAction1-1;
		actionCounts[jamAction1]=actionCounts[jamAction1]+1;	
		actionCounts[jamAction2]=actionCounts[jamAction2]+1;	
	}
//	cout<<"update ";
//	for(int i=0;i<5;i++) cout<<actionCounts[i]<<" ";
//	cout<<endl;
}



int randaction=1;
void Jammer::Random()
{
	_counter++;
	for(int ss=0;ss<_spec.size();ss++) _spec[ss]=0;
	for(int i=0;i<sweepSig.size();i++)
	{
		
		Tone sig=sweepSig[i];

		if(_counter==20) {
			_counter=0;
			for(int kkk=0;kkk<100;kkk++)
			{
				int jamAct=(random() % 5)+1;
				int temp=jamAct * sig.bandWidth-sig.bandWidth/2;
				if(temp!=sweepSig[i].startFreq)
				{
					//cout<<"change "<<sweepSig[i].startFreq<<" "<<temp<<endl;
					sweepSig[i].startFreq=temp;
					randaction=2*jamAct-1;
					//getchar();
					break;
				}
			}
		}
		
		uint32_t _cur_center_freq=sig.startFreq;//+sig.sweepSpeed;	
		for(int ss=0;ss<sig.bandWidth;ss++)
		{
			int destPos=ss+_cur_center_freq-sig.bandWidth/2;
			if((destPos>=0) and (destPos<_spec.size())) _spec[destPos]=sig.spec[ss];
		}
	}
	
	
}

void Jammer::Sweep()
{
	for(int ss=0;ss<_spec.size();ss++) _spec[ss]=0;
	for(int i=0;i<sweepSig.size();i++)
	{
		
		Tone sig=sweepSig[i];
		uint32_t _cur_center_freq=sig.startFreq+sig.sweepSpeed;	
		if (_cur_center_freq>_spec.size()+sig.bandWidth)
		{
			_cur_center_freq=0;
			
		}
		sweepSig[i].startFreq=_cur_center_freq;
		
		for(int ss=0;ss<sig.bandWidth;ss++)
		{
			int destPos=ss+_cur_center_freq-sig.bandWidth/2;
			if((destPos>=0) and (destPos<_spec.size())) _spec[destPos]=sig.spec[ss];
		}
	}
	

	
}

void Jammer::Comb()
{
	for(int ss=0;ss<_spec.size();ss++) _spec[ss]=0;
	for(int i=0;i<combSig.size();i++)
	{
		Tone sig=combSig[i];
		uint32_t _cur_center_freq=sig.startFreq+sig.sweepSpeed;	
		if (_cur_center_freq>_spec.size()+sig.bandWidth)
		{
			_cur_center_freq=0;
			
		}
		sweepSig[i].startFreq=_cur_center_freq;
		
		for(int ss=0;ss<sig.bandWidth;ss++)
		{
			int destPos=ss+_cur_center_freq-sig.bandWidth/2;
			if((destPos>=0) and (destPos<_spec.size())) _spec[destPos]=sig.spec[ss];
		}
	}
	

	
}

void Sort(int a[], int n, int id[], int m)
{
 if ( m > 1)
 {
  int i = 0; 
  int j = m-1;
  int tmp = id[i];
  while(i<j)
  {
       while(j > i && a[id[j]] > a[tmp]) 
            --j;
       if (j > i)
            id[i++] = id[j];  //只改变索引顺序
       while(j > i && a[id[i]] < a[tmp])
            ++i;
       if (j > i)
            id[j--] = id[i];  //只改变索引顺序
  }
  id[i] = tmp;
  Sort(a, n, id, i);
  Sort(a, n, id + i + 1, m - i - 1);
 }
}

int getMax(float data[], int num)
{
	float max=0;
	int index=-1;
	for(int i=0;i<num;i++)
	{
		if(max<=data[i]){
			max=data[i];
			index=i;		
		}
	}
	//cout<<index<<" "<<max<<endl;
	if(index>=0){
		data[index]=-1;
		return index;	
	}
	else return -1;
	
}

void Jammer::getSmartAction()
{
	
	//for(int i=0;i<5;i++) cout<<actionCounts[i]<<" ";
	//cout<<endl;
	
	float tempAction[5];
	memcpy(tempAction,actionCounts,5*sizeof(float));
	for(int i=0;i<3;i++)
	{
		int pos=getMax(tempAction,5);
		curSmartFreq[i]=_band_width*pos+_band_width/2;
		//cout<<"============== pos="<<pos;
	}
	//cout<<endl;

	
}

void Jammer::Smart()
{
	
	
	//if(timePast==10) {
		getSmartAction();
		timePast=0;
	//}
	timePast++;
	
	for(int ss=0;ss<_spec.size();ss++) _spec[ss]=0;
	for(int i=0;i<smartSig.size();i++)
	{
		Tone sig=smartSig[i];
		uint32_t _cur_center_freq=curSmartFreq[i];
				
		for(int ss=0;ss<sig.bandWidth;ss++)
		{
			int destPos=ss+_cur_center_freq-sig.bandWidth/2;
			if((destPos>=0) and (destPos<_spec.size())) _spec[destPos]=sig.spec[ss];
		}
	}
	
	
	
	
}

PlayEngine::PlayEngine()
{
	_ObserveTime=ObserveTime;
	_SpecLen=SpecLen;
	_waterFallImage.create(_ObserveTime,_SpecLen,CV_8UC3);
	_curSpecImage.create(128,_SpecLen,CV_8UC3);
	_center_freq=0;
	_user_band_width=40;

	for(auto tt=0;tt<_ObserveTime;tt++)
	{
		Spectrum sp;
		for(auto ss=0;ss<_SpecLen;ss++) sp[ss]=0;
		_Waterfall.push_back(sp);
	}

	
	
	userSig.bandWidth=_user_band_width;
	userSig.startFreq=_center_freq-_user_band_width/2;
	
	userPower=0;
	for(int ss=0;ss<userSig.bandWidth;ss++)
	{
		uint8_t value=rcosine(ss,userSig.bandWidth,0.5)*(0-FLOOR_DBM)*2;
		userSig.spec.push_back(value);
		userPower=logadd(userPower,value);
	}

	
	generateColorMap();
	
	
}

void PlayEngine::Show()
{
	
	//频谱图
	cv::Point start, end;
	start=cv::Point(0,128);
	int Scale=2;
	_curSpecImage = cv::Mat::zeros(128,_SpecLen*Scale,CV_8UC3);
	for(auto ss=0;ss<_SpecLen;ss++)
	{
		end=cv::Point(ss*Scale,(256-_curSp[ss])/2);
		cv::line(_curSpecImage, start, end, cv::Scalar(0, 255, 0)); 
		start=end;
	}
	int startFreq=_center_freq-_user_band_width/2;
	int stopFreq=_center_freq+_user_band_width/2;
	
	
	start=cv::Point(startFreq*Scale,128);
	for(auto i=startFreq;i<stopFreq;i++)
	{
		end=cv::Point(i*Scale,32);
		cv::line(_curSpecImage, start, end, cv::Scalar(255, 255, 255)); 
		start=end;
	}
	end=cv::Point(stopFreq*Scale,128);
	cv::line(_curSpecImage, start, end, cv::Scalar(255, 255, 255)); 
	
	
	//瀑布图
	for(auto y=0;y<_waterFallImage.rows;y++)
	{
		cv::Vec3b *raw=_waterFallImage.ptr<cv::Vec3b>(y);
		for(auto x=0;x<_waterFallImage.cols;x++)
		{
			WFcolor wf=bytesTovec3(_Waterfall[_waterFallImage.rows-y-1][x]);
			raw[x]=cv::Vec3b(wf.b,wf.g,wf.r);
		}
	}
	
	
	
	
	resize(_waterFallImage,_bigWaterFallImage,cv::Size(SpecLen*2,ObserveTime*2),0,0,cv::INTER_LINEAR);
	cv::imshow("WaterFall",_bigWaterFallImage);
	cv::imshow("Spectrum",_curSpecImage);
	cv::waitKey(20);
}


int PlayEngine::getCurState()
{//
	int pState[5];
	for(int i=0;i<5;i++)
	{
		float power=0;
		for(int j=0;j<40;j++)
		{
			power=power+pow(10,_curSp[i*40+j]/10);
		}
		power=10*log10(power);
		
		if(power>40)
			pState[i]=1;
		else
			pState[i]=0;
	}
	//cout<<"pState=";
	int State=0;
	for(int i=0;i<5;i++)
	{
		//cout<<pState[i]<<" ";
		State=State*2+pState[i];
	}
	//cout<<endl;
//	cout<<State<<endl;
	return State;
	
}


float PlayEngine::Step(int action)
{
	//action=prejamact;

	//添加背景噪声
	for(auto ss=0;ss<_SpecLen;ss++)
	{
		_curSp[ss]=random()%30;
	}
	
	//各角色执行代码
	
	//干扰设备
	jam.Action();
	jam.updateCounts(action);
	

	//通信设备决策中心频点
	_center_freq=_user_band_width/2*(action+1);//action =1,2,3,...,9
	
	
	
	//干扰频谱
	Spectrum *jamSpec=jam.Spec();
	
	for(auto ss=0;ss<_SpecLen;ss++)
	{
		_curSp[ss]=logadd(_curSp[ss],(*jamSpec)[ss]);
	}
	
	
	//计算当前频谱所受到的干扰大小
	int start=_center_freq-_user_band_width/2;
	int stop=_center_freq+_user_band_width/2;
	int noise=0;
	
	for(auto ff=start;ff<stop;ff++)
	{
		noise=logadd(noise,_curSp[ff]);
	}
	
	
	
	//添加用户频谱
	
	for(int ss=0;ss<userSig.bandWidth;ss++)
	{
		int destPos=ss+_center_freq-userSig.bandWidth/2;
		if((destPos>=0) and (destPos<_SpecLen))
			_curSp[destPos]=logadd(_curSp[destPos],userSig.spec[ss]);

	}

	//计算效益
	float reward=0;
	float discount=1.0;
	
	int snr=userPower-noise;


	
	

	if(snr<10) reward=0.0;
	else reward=1.0;
	//std::cout<<"S="<<userPower<<" N="<<noise<<" SNR="<<snr<<" reward="<<reward<<std::endl;
		
	_Waterfall.push_back(_curSp);
	_Waterfall.pop_front();
	

		
	//prejamact=randaction;


	
	return reward;	
}





