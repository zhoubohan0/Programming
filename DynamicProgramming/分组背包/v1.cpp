//�汾1
#include<iostream>
using namespace std;
 
int dp[1005]={0};
int m[101][101]={0},v[101][101]={0},length[1001]={0};
int main()
{
	int M,n,a,b,c,group=0;
	cin>>M>>n;
	//���鱳�������ȴ����� 
	for(int i=1;i<=n;i++){
		cin>>a>>b>>c;
		group = max(group,c);//֪�����м��� 
		m[c][++length[c]]=a;
		v[c][length[c]]=b;
	}
	for(int k=1;k<=group;k++)//ÿ���ڿ���01���������ע��һ���������ѭ�� 
		for(int j=M;j>=0;j--)//ͬ01������������� 
			for(int num=1;num<=length[k];num++)//ÿ�� 
				if(j-1*m[k][num]>=0)//��ʱ��ÿ��ÿ������Сֵ��֪������Ҫ�ж� 
					dp[j] = max(1*v[k][num]+dp[j-1*m[k][num]],dp[j]);
	cout<<dp[M];
	return 0;
}
 
