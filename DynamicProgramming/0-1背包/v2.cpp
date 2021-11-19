//°æ±¾2
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
			for(int j=time;j>=bin*t;j--) 
				dp[j] = max(bin*v+dp[j-bin*t],dp[j]);
	}
	cout<<dp[time];
	return 0;
}
 
