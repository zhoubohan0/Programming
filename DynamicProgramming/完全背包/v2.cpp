//�汾2
#include<iostream>
using namespace std;
long long dp[10000005]={0};
int main()
{
	int time,n,t,v;
	cin>>time>>n;
	for(int i=1;i<=n;i++){
		cin>>t>>v;
		for(int j=1*t;j<=time;j++)//01����������ı�������ǲ����ظ����£������ر����������Ҫ׷���ظ����£�װ��һ������װһ����װһ��װһ�������᲻���������ļ�ֵ 
			dp[j] = max(1*v+dp[j-1*t],dp[j]);
	}
	cout<<dp[time];
	return 0;
}
 
