uniform sampler2D depthTex;

void main()
{
	//make this a uniform variable;
	float maxDepth = 0.5;
	float depth = texture2D(depthTex,TexCoord[0].st).x;
	if(depth>maxDepth)
	{
		discard;
		return;
	}	

}
