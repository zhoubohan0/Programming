//°æ±¾3
#include<iostream>
using namespace std;
int dp[1005]={0};
int main()
{
	int time,n,t,v;
	cin>>time>>n;
	for(int i=1;i<=n;i++){
		cin>>t>>v;
		for(int j=time;j>=1*t;j--) 
			dp[j] = max(1*v+dp[j-1*t],dp[j]);
	}
	cout<<dp[time];
	return 0;
}
 
