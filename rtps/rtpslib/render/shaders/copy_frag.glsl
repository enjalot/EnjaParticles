uniform sampler2D normalTex;
uniform sampler2D depthTex;

void main(void)
{
    float fldepth = texture2D(depthTex, gl_TexCoord[0].xy).x;
    if(fldepth > .999)
    {
        discard;
    }    
    gl_FragColor = texture2D(normalTex, gl_TexCoord[0].xy);
    gl_FragDepth = fldepth; 

}
