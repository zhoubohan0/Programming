//°æ±¾1:TLE
#include<iostream>
using namespace std;
int dp[10000005]={0};
int main()
{
	int time,n,t,v;
	cin>>time>>n;
	for(int i=1;i<=n;i++){
		cin>>t>>v;
		for(int num=0;num<=time/float(t);num++)
			for(int j=time;j>=num*t;j--) 
				dp[j] = max(num*v+dp[j-num*t],dp[j]);
	}
	cout<<dp[time];
	return 0;
}
 
