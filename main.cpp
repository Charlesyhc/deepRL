#include "PlayEngine.hpp"
#include "dqn.hpp"




extern int randaction;

double CalculateEpsilon(const int iter);
double PlayOneEpisode(PlayEngine &engine, dqn::DQN& Dqn,const double epsilon,const bool update); 

void SaveReward(float reward)
{
	FILE *fp=fopen("record.txt","a+");
	if(fp!=NULL)
	{
		fprintf(fp,"%f\n",reward);
		fclose(fp);
	}
}
using namespace std;





int main()
{
	PlayEngine engine;
	
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	
	dqn::ActionVect legal_actions;
	for(uint8_t i=0;i<ActionNum;i++) legal_actions.push_back(i); 
													//memeory  Discount factor of future rewards (0,1]	
	dqn::DQN Dqn(legal_actions, "dqn_solver.prototxt", Memory_Size, Gamma);
	Dqn.Initialize();
	
	
	remove("record.txt");
	//训练模式
	for (auto episode = 0;; episode++)
	{
		std::cout << "episode: " << episode << std::endl;
		const auto epsilon = CalculateEpsilon(Dqn.current_iteration());
		PlayOneEpisode(engine,Dqn, epsilon, true);
	}
}


dqn::Action LastAction=0;
double PlayOneEpisode(PlayEngine &engine, dqn::DQN& Dqn,const double epsilon,const bool update)
{
	std::deque<dqn::FrameDataSp> past_frames;

	
	//获取当前感知的瀑布图
	const auto current_frame = dqn::PreprocessScreen(engine.Waterfall());
	
	dqn::InputFrames input_frames=current_frame;
	
	//根据当前输入确定策略 
	const auto action = Dqn.SelectAction(input_frames, epsilon);
	
	//执行策略 计算回报
	float reward =0.0;
	for(int i=0;i<STEP_OVER;i++)
	{	
		reward=reward+engine.Step((int)action);
		engine.Show();
	}
	std::cout<<"raw reward="<<reward<<std::endl;
	if (reward==STEP_OVER) reward=1;else reward=0;
	float discount;
	if(LastAction!=action) discount=0.8;else discount=1;
	LastAction=action;
	reward=reward*discount;
	std::cout<<"current reward="<<reward<<std::endl<<std::endl;
	
	//save reward
	SaveReward(reward);

	
	
	//训练阶段，则需要update网络
	if(update)
	{
		const auto transition = 
			dqn::Transition(input_frames,action,reward,dqn::PreprocessScreen(engine.Waterfall()));
		Dqn.AddTransition(transition);
		// If the size of replay memory is enough, update DQN
		if (Dqn.memory_size() > Memory_TH) Dqn.Update();
	}
	
	
	
	
	
}





//选择随机策略的概率
float greed=0.95;
double CalculateEpsilon(const int iter) {
  if (iter < Iter_Explore) {
    return 1.0 - greed * (static_cast<double>(iter) / Iter_Explore); //Iter_Explore 次迭代后 取得随机策略的概率为0.1
  } else {
    return 1-greed;
  }
}
	



