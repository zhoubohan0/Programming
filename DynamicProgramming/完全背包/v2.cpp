//版本2
#include<iostream>
using namespace std;
long long dp[10000005]={0};
int main()
{
	int time,n,t,v;
	cin>>time>>n;
	for(int i=1;i<=n;i++){
		cin>>t>>v;
		for(int j=1*t;j<=time;j++)//01背包中这里的本质理解是不能重复更新，而多重背包这里就是要追求重复更新：装了一件后再装一件再装一件装一件看看会不会产生更大的价值 
			dp[j] = max(1*v+dp[j-1*t],dp[j]);
	}
	cout<<dp[time];
	return 0;
}
 
