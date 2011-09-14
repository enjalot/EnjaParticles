/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#define STRINGIFY(A) #A

//this code heavily based off NVIDIA's oclParticle example from the OpenCL SDK

const char* vertex_shader_source = STRINGIFY(

                                            uniform float pointRadius;  // point size in world space
                                            uniform float pointScale;   // scale to calculate size in pixels
                                            uniform bool blending;
                                            //uniform float densityScale;
                                            //uniform float densityOffset;
                                            //varying float pointRadius;
                                            varying vec3 posEye;        // position of center in eye space


                                            void main(){

                                            //posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
                                            //float dist = length(posEye);
                                            //we packed radius in the 4th component of vertex
                                            //pointRadius = gl_Vertex.w;
                                            //gl_PointSize = pointRadius * (pointScale / dist);
                                            gl_PointSize = 20.0;
                                            //gl_PointSize = pointRadius * (1.0 / dist);

                                            gl_TexCoord[0] = gl_MultiTexCoord0;
                                            //gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
                                            gl_Position = ftransform();
                                            gl_Position.z = gl_Position.z*gl_Position.w/5.0;

                                            /*gl_FrontColor = gl_Color;
                                        
                                            if(!blending)
                                            {
                                                gl_FrontColor.w = 1.0;
                                            }*/



                                            }


                                            );

const char* fragment_shader_source = STRINGIFY(

                                              uniform float pointRadius;  // point size in world space
                                              //varying float pointRadius;  // point size in world space
                                              varying vec3 posEye;        // position of center in eye space
                                              //uniform sampler2D col;

                                              void main(){
                                              /*const vec3 lightDir = vec3(0.577, 0.577, 0.577);
                                              const float shininess = 40.0;
                                          
                                              // calculate normal from texture coordinates
                                              vec3 n;
                                              n.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
                                              float mag = dot(n.xy, n.xy);
                                              if (mag > 1.0) discard;   // kill pixels outside circle
                                              n.z = sqrt(1.0-mag);
                                          
                                              // point on surface of sphere in eye space
                                              vec3 spherePosEye = posEye + n*pointRadius;
                                              //vec3 spherePosEye = posEye + n*pointRadius;
                                          
                                              // calculate lighting
                                              float diffuse = max(0.0, dot(lightDir, n));
                                              
                                              vec3 v = normalize(-spherePosEye);
                                              vec3 h = normalize(lightDir + v);
                                              float specular = pow(max(0.0, dot(n, h)), shininess); */


                                              gl_FragColor = gl_Color * diffuse + specular;
                                              /*
                                              vec4 tex = texture2D(col, gl_TexCoord[0].st);
                                              gl_FragColor = tex;
                                              //gl_FragColor.w = gl_Color.w * (tex.x + tex.y + tex.z)/3.;
                                              gl_FragColor.w = (tex.x + tex.y + tex.z)/3.;
                                              */
                                              }


                                              );


