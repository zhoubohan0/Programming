//版本1
#include<iostream>
using namespace std;
 
int dp[1005]={0};
int m[101][101]={0},v[101][101]={0},length[1001]={0};
int main()
{
	int M,n,a,b,c,group=0;
	cin>>M>>n;
	//分组背包必须先存再算 
	for(int i=1;i<=n;i++){
		cin>>a>>b>>c;
		group = max(group,c);//知道共有几组 
		m[c][++length[c]]=a;
		v[c][length[c]]=b;
	}
	for(int k=1;k<=group;k++)//每组内看作01背包，因此注组一定是最外层循环 
		for(int j=M;j>=0;j--)//同01背包，反向更新 
			for(int num=1;num<=length[k];num++)//每件 
				if(j-1*m[k][num]>=0)//这时候每组每件的最小值不知道所以要判断 
					dp[j] = max(1*v[k][num]+dp[j-1*m[k][num]],dp[j]);
	cout<<dp[M];
	return 0;
}
 
