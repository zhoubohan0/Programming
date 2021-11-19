//版本1
#include<iostream>
using namespace std;
int dp[1005]={0};
int main()
{
	int time,n,t,v;
	cin>>time>>n;
	for(int i=1;i<=n;i++){
		cin>>t>>v;
		for(int bin=0;bin<=1;bin++)
			for(int j=time;j>=0;j--)//一定从后向前！ 
				if(j-bin*t>=0&&dp[j]<bin*v+dp[j-bin*t])
					dp[j] = bin*v+dp[j-bin*t];
	}
	cout<<dp[time];
	return 0;
}
 
