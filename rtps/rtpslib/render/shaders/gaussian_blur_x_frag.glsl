uniform sampler2D depth; // the texture with the scene you want to blur
 
//const float blurSize = 1.0/512.0; // I've chosen this size because this will result in that every step will be one pixel wide if the depth texture is of size 512x512
 
void main(void)
{
   float sum = 0.0;
   float blurSize = dFdx(gl_TexCoord[0]);
   // blur in y (vertical)
   // take nine samples, with the distance blurSize between them
   sum += texture2D(depth, vec2(gl_TexCoord[0].x - 4.0*blurSize, gl_TexCoord[0].y)).x * 0.05;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x - 3.0*blurSize, gl_TexCoord[0].y)).x * 0.09;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x - 2.0*blurSize, gl_TexCoord[0].y)).x * 0.12;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x - blurSize, gl_TexCoord[0].y)).x * 0.15;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x, gl_TexCoord[0].y)).x * 0.16;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x + blurSize, gl_TexCoord[0].y)).x * 0.15;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x + 2.0*blurSize, gl_TexCoord[0].y)).x * 0.12;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x + 3.0*blurSize, gl_TexCoord[0].y)).x * 0.09;
   sum += texture2D(depth, vec2(gl_TexCoord[0].x + 4.0*blurSize, gl_TexCoord[0].y)).x * 0.05;
 
   gl_FragData[0] = vec4(sum,sum,sum,1.0);
   gl_FragDepth = sum;
}
