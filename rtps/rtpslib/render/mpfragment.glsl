#version 330
//#fragment shader
in vec2 texCoord;
out vec4 outColor;

uniform sampler2D col;
uniform float emit, alpha;

void main(void) {

    // load color texture
    vec4 color = vec4(1.,0.,0.,1.);
    color = texture2D(col, texCoord);
    color.a = 1*(color.r + color.g + color.b)/3.;

    // apply material panel values
    //color.rgb *= emit;
    //color.a *= alpha;
    //color.a = magnitude(vec3(color.r, color.g, color.b));

    outColor = color;
}
