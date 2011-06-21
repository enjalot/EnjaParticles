uniform sampler2D depthTex; 
//uniform float width;
//uniform float height;
uniform float del_x;
uniform float del_y;
uniform float h_x;
uniform float h_y;
//uniform float focal_x;
//uniform float focal_y;
uniform float dt;
uniform float distance_threshold;

float secondOrderCenterDifference(vec2 texCoord,float depth, vec2 dir, float h)
{
    if(abs(texture2D(depthTex, texCoord+dir).x- 2.0*depth +texture2D(depthTex, texCoord-dir).x)>distance_threshold)
        return 0.0;
    return (texture2D(depthTex, texCoord+dir).x- 2.0*depth +texture2D(depthTex, texCoord-dir).x)/(h*h);
}

float centerDifference(vec2 texCoord, vec2 dir, float h)
{
    if(abs(texture2D(depthTex, texCoord+dir).x-texture2D(depthTex, texCoord-dir).x)>distance_threshold)
        return 0.0;
    return (texture2D(depthTex, texCoord+dir).x-texture2D(depthTex, texCoord-dir).x)/(2.0*h);
}

float forwardDifference(float fxh, float fx, float h)
{
    return (fxh-fx)/h;
}

void main()
{
    vec2 dx_texCoord = gl_TexCoord[0].st+vec2(del_x,0.0);
    vec2 dy_texCoord = gl_TexCoord[0].st+vec2(0.0,del_y);
    float Cx = gl_ProjectionMatrix[0].x;//2.0/(width*focal_x);
    float Cy = gl_ProjectionMatrix[1].y;//2.0/(height*focal_y);
    float Cx2 = Cx*Cx;
    float Cy2 = Cy*Cy;
    float depth=texture2D(depthTex, gl_TexCoord[0].st).x;	
    float dx = centerDifference(gl_TexCoord[0].st,vec2(del_x,0.0),h_x);
    float ddx = secondOrderCenterDifference(gl_TexCoord[0].st,depth,vec2(del_x,0.0),h_x);
    float dy = centerDifference(gl_TexCoord[0].st,vec2(0.0,del_y),h_y);
    float ddy = secondOrderCenterDifference(gl_TexCoord[0].st,depth,vec2(0.0,del_y),h_y);
    float D = (Cy2*dx*dx)+(Cx2*dy*dy)+(Cy2*Cx2*depth*depth);
    float depthX = texture2D(depthTex, dx_texCoord).x;
    float dxdx = centerDifference(dx_texCoord,vec2(del_x,0.0),h_x);
    float dydx = centerDifference(dx_texCoord,vec2(0.0,del_y),h_y);
    float depthY = texture2D(depthTex, dy_texCoord).x;
    float dxdy = centerDifference(dy_texCoord,vec2(del_x,0.0),h_x);
    float dydy = centerDifference(dy_texCoord,vec2(0.0,del_y),h_y);
    float Dx=(Cy2*dxdx*dxdx)+(Cx2*dydx*dydx)+(Cy2*Cx2*depthX*depthX);
    float Dy=(Cy2*dxdy*dxdy)+(Cx2*dxdy*dxdy)+(Cy2*Cx2*depthY*depthY);
    float dDx = forwardDifference(Dx,D,h_x);
    float dDy = forwardDifference(Dy,D,h_y);
    float Ex = 0.5*dx*dDx-ddx*D;
    float Ey = 0.5*dy*dDy-ddy*D;
    float H = (Cy*Ex+Cx*Ey)/(pow(D,1.5)*2.);
    gl_FragDepth = depth + dt*H;

}
