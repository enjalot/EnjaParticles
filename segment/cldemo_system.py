"""
This system is for visualizing the inner workings of SPH

"""


#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush, glGenBuffers, glBindBuffer
from OpenGL.GL import *
from OpenGL.GL.ARB.geometry_shader4 import *

import numpy
import pyopencl as cl
import glutil
import util
timings = util.timings

from clsystem import CLSystem 
import clpermute
import cldensity
import cldiffuse 
import clforce
import clcollision_wall
import clleapfrog
import clleapfrog_diffuse
import clghost_density
import clghost_force
import cldemo_update


class CLDemoSystem(CLSystem):
    def __init__(self, sph, dt=.001, is_ghost=False, ghost_system=None):
        CLSystem.__init__(self, sph, dt, is_ghost, ghost_system)
       
        self.with_ghost_density = True
        self.with_ghost_force = True
 
        self.density = cldensity.CLDensity(self)
        self.diffuse = cldiffuse.CLDiffuse(self)
        self.force = clforce.CLForce(self)
        self.collision_wall = clcollision_wall.CLCollisionWall(self)
        #self.leapfrog = clleapfrog.CLLeapFrog(self)
        self.leapfrog = clleapfrog_diffuse.CLLeapFrogDiffuse(self)
        self.ghost_density = clghost_density.CLGhostDensity(self)
        self.ghost_force = clghost_force.CLGhostForce(self)
        
        self.demo_update = cldemo_update.CLDemoUpdate(self)

        self.random_inc = 0
        self.target_intensity = 1.0

        #self.init_shaders()
        self.init_shadersGL3()
 
       
    
    def acquire_gl(self):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)
    def release_gl(self):
        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)


    def update(self):
        self.acquire_gl()


        numpy.set_printoptions(precision=6, linewidth=1000)
        """
        pos = numpy.ndarray((self.num,4), dtype=numpy.float32)
        cl.enqueue_read_buffer(self.queue, self.position_u, pos)
        print "pos_u at begining"
        print pos.T[0:100] 
        """

        self.exec_hash()

        """
        print "hashes before"
        hashes = numpy.ndarray((self.system.max_num,), dtype=numpy.uint32)
        cl.enqueue_read_buffer(self.queue, self.sort_hashes, hashes).wait()
        print hashes[0:100].T

        print "indices before"
        indices = numpy.ndarray((self.system.max_num,), dtype=numpy.uint32)
        cl.enqueue_read_buffer(self.queue, self.sort_indices, indices).wait()
        print indices[0:100].T
        """
        #self.queue.finish()

        if self.num > 0:
            self.exec_sort()
 
        """
        print "hashes after"
        #hashes = numpy.ndarray((self.system.max_num,), dtype=numpy.int32)
        cl.enqueue_read_buffer(self.queue, self.sort_hashes, hashes).wait()
        print hashes[0:100].T

        print "indices after"
        #indices = numpy.ndarray((self.system.max_num,), dtype=numpy.int32)
        cl.enqueue_read_buffer(self.queue, self.sort_indices, indices).wait()
        print indices[0:100].T
        """



        negone = numpy.ones((self.system.domain.nb_cells+1,), dtype=numpy.int32)
        negone *= -1
        cl.enqueue_write_buffer(self.queue, self.ci_start, negone)

        cl.enqueue_write_buffer(self.queue, self.ci_end, numpy.zeros((self.system.domain.nb_cells+1), dtype=numpy.int32))
        self.queue.finish()

        self.exec_cellindices()


        """
        tmp_start = numpy.ndarray((self.system.domain.nb_cells,), dtype=numpy.uint32)
        tmp_end = numpy.ndarray((self.system.domain.nb_cells,), dtype=numpy.uint32)
        cl.enqueue_read_buffer(self.queue, self.ci_start, tmp_start)
        cl.enqueue_read_buffer(self.queue, self.ci_end, tmp_end)
        import sys
        for i in xrange(len(tmp_start)):
            if tmp_start[i] <= self.system.domain.nb_cells+1 or tmp_end[i] > 0:
                print i, tmp_start[i], tmp_end[i]
        """

        self.exec_permute()
        
        if not self.is_ghost:
        #if True:
            self.exec_density()
            
            #self.exec_diffuse()
            """
            color = numpy.ndarray((self.num,4), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.color_s, color)
            print color.T
            """
             
            if self.ghost_system is not None and self.with_ghost_density:
                self.exec_ghost_density()



            """
            density = numpy.ndarray((self.num,), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.density_s, density)
            print "density"
            print density.T


            gdensity = numpy.ndarray((self.num,), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.ghost_density_s, gdensity)
            print gdensity.T


            clf = numpy.ndarray((self.num,4), dtype=numpy.float32)
            cl.enqueue_read_buffer(self.queue, self.clf_debug, clf)
            print clf
            cl.enqueue_read_buffer(self.queue, self.xsph_s, pos)
            print "xpsh_s before force"
            print pos.T[0:100] 
            """
 
            self.exec_force()

            """
            if self.num > 0:
                clf = numpy.ndarray((self.num,4), dtype=numpy.float32)
                cl.enqueue_read_buffer(self.queue, self.clf_debug, clf)
                print "clf"
                print clf[0:100][0]
            """
     


            if self.ghost_system is not None and self.with_ghost_force:
                self.exec_ghost_force()

                """
                force = numpy.ndarray((self.num,4), dtype=numpy.float32)
                cl.enqueue_read_buffer(self.queue, self.force_s, force)
                print force.T[0:100] 


                clf = numpy.ndarray((self.num,4), dtype=numpy.float32)
                cl.enqueue_read_buffer(self.queue, self.clf_debug, clf)
                print clf.T
                """
     
 
            """
            cl.enqueue_read_buffer(self.queue, self.force_s, pos)
            print "force_s after force"
            print pos.T[0:100] 
            cl.enqueue_read_buffer(self.queue, self.position_u, pos)
            print "position_u before leapfrog"
            print pos.T[0:100] 
            print "num", self.num
            cl.enqueue_read_buffer(self.queue, self.velocity_s, pos)
            print "velocity_s before leapfrog"
            print pos.T[0:100] 

            """

            self.exec_collision_wall()

           
            self.exec_leapfrog()

            self.exec_demo_update()

            """
            cl.enqueue_read_buffer(self.queue, self.position_u, pos)
            print "position_u after leapfrog"
            print pos.T[0:100] 
            """
            """
            if self.num > 0:
                col = numpy.ndarray((self.num,4), dtype=numpy.float32)
                cl.enqueue_read_buffer(self.queue, self.color_u, col)
                print "color_u after demo"
                print col[0:5].T 
            """
 

 

            #"""                


        self.release_gl()
        

    @timings("Permute")
    def exec_permute(self):
        self.permute.execute(   self.num, 
                                self.position_u,
                                self.position_s,
                                self.velocity_u,
                                self.velocity_s,
                                self.veleval_u,
                                self.veleval_s,
                                self.color_u,
                                self.color_s,
                                self.sort_indices
                                #self.clf_debug,
                                #self.cli_debug
                            )

    @timings("Density")
    def exec_density(self):
        self.density.execute(   self.num, 
                                self.position_s,
                                self.density_s,
                                self.ci_start,
                                self.ci_end,
                                #self.gp,
                                self.gp_scaled,
                                self.systemp,
                                self.clf_debug,
                                self.cli_debug
                            )

    @timings("Diffuse")
    def exec_diffuse(self):
        for i in xrange(3):
            dt = self.dt * .1
            self.diffuse.execute(   self.num, 
                                    self.position_s,
                                    self.density_s,
                                    self.color_s,
                                    numpy.float32(dt),
                                    self.ci_start,
                                    self.ci_end,
                                    #self.gp,
                                    self.gp_scaled,
                                    self.systemp,
                                    self.clf_debug,
                                    self.cli_debug
                                )




    @timings("Ghost Density")
    def exec_ghost_density(self):
        self.ghost_density.execute( self.num, 
                                    self.position_s,
                                    self.ghost_system.position_s,
                                    self.density_s,
                                    self.ghost_density_s,
                                    self.ghost_system.color_s,
                                    self.systemp,
                                    numpy.float32(self.target_intensity),
                                    self.ghost_system.ci_start,
                                    self.ghost_system.ci_end,
                                    self.ghost_system.gp_scaled,
                                    self.ghost_system.sphp,
                                    self.clf_debug,
                                    self.cli_debug
                                )


    @timings("Force")
    def exec_force(self):
        self.random_inc += 1
        self.force.execute(   self.num, 
                              self.position_s,
                              self.density_s,
                              self.veleval_s,
                              self.force_s,
                              self.xsph_s,
                              numpy.int32(self.random_inc),
                              self.ci_start,
                              self.ci_end,
                              self.gp_scaled,
                              self.systemp,
                              self.clf_debug,
                              self.cli_debug
                          )

    @timings("Ghost Force")
    def exec_ghost_force(self):
        self.ghost_force.execute(   self.num, 
                                    self.position_s,
                                    self.ghost_system.position_s,
                                    self.density_s,
                                    self.ghost_density_s,
                                    self.ghost_system.color_s,
                                    self.veleval_s,
                                    self.force_s,
                                    self.xsph_s,
                                    self.systemp,
                                    numpy.float32(self.target_intensity),
                                    self.ghost_system.ci_start,
                                    self.ghost_system.ci_end,
                                    self.ghost_system.gp_scaled,
                                    self.ghost_system.sphp,
                                    self.clf_debug,
                                    self.cli_debug
                )

    @timings("Collision Wall")
    def exec_collision_wall(self):
        self.collision_wall.execute(  self.num, 
                                      self.position_s,
                                      self.velocity_s,
                                      self.force_s,
                                      self.gp_scaled,
                                      self.systemp,
                                      #self.clf_debug,
                                      #self.cli_debug
                                   )



    @timings("Leapfrog")
    def exec_leapfrog(self):
        self.leapfrog.execute(    self.num, 
                                  self.position_u,
                                  self.position_s,
                                  self.velocity_u,
                                  self.velocity_s,
                                  self.color_u,
                                  self.color_s,
                                  self.veleval_u,
                                  self.force_s,
                                  self.xsph_s,
                                  self.sort_indices,
                                  self.systemp,
                                  #self.clf_debug,
                                  #self.cli_debug
                                  numpy.float32(self.dt)
                             )

    @timings("Demo Update")
    def exec_demo_update(self):
        self.demo_update.execute( self.num, 
                                  self.velocity_u,
                                  self.color_u,
                                  self.force_s,
                                  self.density_s,
                                  self.sort_indices,
                                  self.systemp,
                                  #self.clf_debug,
                                  #self.cli_debug
                                  numpy.float32(self.dt)
                             )



    def render(self):
        self.draw()
        self.draw()

    def draw(self):
        if self.num <= 0:
            return
        gc = self.global_color
        glColor4f(gc[0],gc[1], gc[2],gc[3])


        #print glGetString(GL_VERSION)
        radius = self.system.smoothing_radius / self.system.sim_scale

        if self.arrow_pass:
            glUseProgram(self.program_arrow)
            glUniform1f(glGetUniformLocation(self.program_arrow, "radius"), radius)
            glUniform1f(glGetUniformLocation(self.program_arrow, "speed_limit"), self.system.velocity_limit)
        else:
            glUseProgram(self.program)
            glUniform1f(glGetUniformLocation(self.program, "radius"), radius)
            glUniform1f(glGetUniformLocation(self.program, "speed_limit"), self.system.velocity_limit)
        self.arrow_pass = not self.arrow_pass

        #glEnable(GL_POINT_SPRITE);
        #glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
        #glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        #glUseProgram(glsl_program[SPHERE_SHADER]);
        #float particle_radius = 0.125f * 0.5f;
        #glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointScale"), ((float)window_width) / tanf(65. * (0.5f * 3.1415926535f/180.0f)));
        #//glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointRadius"), particle_radius );
        #glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointRadius"), particle_radius*radius_scale ); //GE
        #glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "near"), near_depth );
        #glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "far"), far_depth );


        #glEnable(GL_POINT_SMOOTH)
        #glPointSize(15)

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable(GL_DEPTH_TEST)
        glDisable(GL_DEPTH_TEST)
        #glDepthMask(GL_FALSE)

        """
        glColor3f(1., 0, 0)
        glBegin(GL_POINTS)
        for p in self.pos_vbo.data:
            glVertex3f(p[0], p[1], p[2])

        glEnd()
        """

        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, None)

        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, None)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glDrawArrays(GL_POINTS, 0, self.num)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        #glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)

    def init_shadersGL3(self):
        #from OpenGL.GL.shaders import *
        from glutil import compileShader#, compileProgram
        vertexsource = """
            #version 120

            //in vec4 vertex;

            varying vec4 geom_color;
            
            //uniform float timer;

            void main() 
            {
                //vec4 v = vertex;
                vec4 v = gl_Vertex;
                //v.z = sin(timer+v.y+v.x)*0.5+v.z;
                //v.x = sin(timer*10.0+v.y+v.y)*0.2+v.x;
                gl_Position = gl_ModelViewProjectionMatrix * v;
                geom_color = vec4(1., 1., 1., 1.);//gl_Color;
                geom_color = gl_Color;
            }
        """
        geometry_source = """
        #version 330
        //#version 120
        //#extension GL_EXT_geometry_shader4 : enable
        //layout(points) in;
        //layout(points) out;


        void main(void)
        {
            gl_Position = gl_in[0].gl_Position;
            //gl_Position = gl_PositionIn[0];
            EmitVertex();
            EndPrimitive();
        }
        """
        geometrysource = """
            #version 330
            //#version 150
            //#extension GL_EXT_geometry_shader4 : enable

            layout(points) in;
            layout(triangle_strip) out;

            out vec2 texCoord;

            in vec4 geom_color[1];

            out vec4 frag_color;

            //#define radius 0.1
            uniform float radius;
            #define layer 1

            void main(void) 
            {
                //for(int i = 0; i < gl_in.length(); i++) {  // avoid duplicate draw
                int j = 0;
                vec4 p = gl_in[0].gl_Position;

                //for (int j=0; j < gl_in.length(); j++) 
                {
                    texCoord = vec2(1.0,1.0);
                    gl_Position = vec4(p.r+radius, p.g+radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();

                    texCoord = vec2(0.0,1.0);
                    gl_Position = vec4(p.r-radius, p.g+radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();

                    texCoord = vec2(1.0,0.0);
                    gl_Position = vec4(p.r+radius, p.g-radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();

                    texCoord = vec2(0.0,0.0);
                    gl_Position = vec4(p.r-radius, p.g-radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();

                    EndPrimitive();
                }
            //}
            }



        """
 
        geometrysource_arrow = """
            #version 330

            //#version 150
            //#extension GL_EXT_geometry_shader4 : enable

            layout(points) in;
            layout(triangle_strip) out;

            out vec2 texCoord;

            in vec4 geom_color[1];

            out vec4 frag_color;

            //#define radius 0.1
            uniform float radius;
            uniform float speed_limit;

            void main(void) 
            {
                //for(int i = 0; i < gl_in.length(); i++) {  // avoid duplicate draw
                int j = 0;
                vec4 p = gl_in[0].gl_Position;

                //for (int j=0; j < gl_in.length(); j++) 
                {

/*
                    texCoord = vec2(1.0,1.0);
                    gl_Position = vec4(p.r+radius, p.g+radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();

                    texCoord = vec2(0.0,1.0);
                    gl_Position = vec4(p.r-radius, p.g+radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();

                    texCoord = vec2(1.0,0.0);
                    gl_Position = vec4(p.r+radius, p.g-radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();

                    texCoord = vec2(0.0,0.0);
                    gl_Position = vec4(p.r-radius, p.g-radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();
                    
                    //arrow
                    texCoord = vec2(1.0,1.0);
                    gl_Position = vec4(p.r+radius, p.g+radius+j*0.05, p.b, p.a);
                    frag_color = geom_color[0];
                    EmitVertex();


*/

                    texCoord = vec2(1.0,1.0);
                    float x = 0.;//-.05;
                    float y = 0.;
                    vec4 base = vec4(p.r + x, p.g + y, p.b, p.a); 
                    texCoord = vec2(1.0,1.0);
                    gl_Position = base;
                    //frag_color = vec4( 0., 1., 0., 1.);
                    frag_color = geom_color[0];
                    EmitVertex();
 
                    texCoord = vec2(.0,.0);
                    base.x += .005;
                    base.y += .005;
                    gl_Position = base;
                    //frag_color = vec4( 0., 1., 0., 1.);
                    frag_color = geom_color[0];
                    EmitVertex();


/*
                    texCoord = vec2(1.0,.0);
                    base.x += .05;
                    //base.y -= .05;
                    gl_Position = base;
                    frag_color = vec4( 0., 1., 0., 1.);
                    EmitVertex();
*/

                    texCoord = vec2(.0,1.0);
                    base.x += geom_color[0].x / speed_limit * .3;
                    base.y += geom_color[0].y / speed_limit * .3;
                    //base.x += .05;
                    //base.y += -.1;
                    gl_Position = base;
                    //frag_color = vec4( 0., 1., 0., 1.);
                    frag_color = geom_color[0];
                    EmitVertex();
                   






                    EndPrimitive();
                }
            //}
            }



        """
       
        fragmentsource = """
            #version 330
            in vec2 texCoord;
            out vec4 outColor;

            in vec4 frag_color;
            uniform float speed_limit;

            //uniform sampler2D col;

            void main(void) 
            {

                vec3 n;
                //n.xy = gl_PointCoord.st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
                n.xy = texCoord*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
                
                float mag = dot(n.xy, n.xy);

                //normalized density
                float dnorm = frag_color.w;

                //if (mag > 1. - dnorm * .5) discard;   // kill pixels outside circle
                if (mag > 1.) discard;   // kill pixels outside circle
               
                if (mag > 1.95 && mag < 1.)
                {
                    vec4 color = vec4(0.,.2,0.,1.);
                    outColor = color;
                }
                else
                {

                    // load color texture
                    //color = texture2D(col, texCoord);
                    
                    //float snorm = frag_color.x;
                    //float snorm = dot(frag_color.xyz, frag_color.xyz) / speed_limit;
                    float snorm = length(frag_color.xy) / speed_limit * 15.;
                    vec4 color = vec4(snorm, 0., 1. - snorm, 1.);
                    //vec4 color = vec4(0., 0., 1., 1.);
                    color *= dnorm  * .1;
                    //vec4 color = vec4(0., frag_color.x, frag_color.w, 1.);

                    outColor = color;
                    //outColor = frag_color;
                }
            }
        """ 
        fragmentsource_arrow = """
            #version 330
            in vec2 texCoord;
            out vec4 outColor;

            in vec4 frag_color;

            uniform float speed_limit;
            //uniform sampler2D col;

            void main(void) 
            {

                vec3 n;
                //n.xy = gl_PointCoord.st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
                n.xy = texCoord*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
                
                float mag = dot(n.xy, n.xy);

                //if (mag > 1.) discard;   // kill pixels outside circle
               
               /*
                if (mag > .95 && mag < 1.)
                {
                    vec4 color = vec4(0.,.1,0.,1.);
                    outColor = color;
                }
                else
                {
               */ 

                    // load color texture
                    //color = texture2D(col, texCoord);
                    
                    //float snorm = frag_color.x;
                    //float snorm = dot(frag_color.xy, frag_color.xy) / speed_limit * 5.;
                    float snorm = length(frag_color.xy) / speed_limit * 10.;
                    float dnorm = frag_color.w;
                    vec4 color = vec4(snorm, 0., 1. - snorm, 1.);
                    //color *= dnorm  * .1;
                    //vec4 color = vec4(0., 0.7, 0., 1.);
                    //vec4 color = vec4(0., frag_color.x, frag_color.w, 1.);

                    outColor = color;
                    //outColor = frag_color;
               // }
            }
        """ 


        vshader = compileShader( vertexsource, GL_VERTEX_SHADER )
        fshader = compileShader( fragmentsource, GL_FRAGMENT_SHADER )
        #fshader = compileShader( fragment_source, GL_FRAGMENT_SHADER )
        gshader = compileShader( geometrysource, GL_GEOMETRY_SHADER )
        #gshader = compileShader( geometry_source, GL_GEOMETRY_SHADER )
        #self.program = compileProgram( vshader, fshader)
 
        self.program = glCreateProgram()
        glProgramParameteriARB(self.program, GL_GEOMETRY_INPUT_TYPE_ARB, GL_POINTS)
        glProgramParameteriARB(self.program, GL_GEOMETRY_OUTPUT_TYPE_ARB, GL_POINTS) #not sure why this works
        glProgramParameteriARB(self.program, GL_GEOMETRY_VERTICES_OUT_ARB, 200)


        self.compileProgram( self.program, vshader, fshader, gshader)

        vshader = compileShader( vertexsource, GL_VERTEX_SHADER )
        fshader = compileShader( fragmentsource_arrow, GL_FRAGMENT_SHADER )
        gshader = compileShader( geometrysource_arrow, GL_GEOMETRY_SHADER )
        self.program_arrow = glCreateProgram()
        glProgramParameteriARB(self.program_arrow, GL_GEOMETRY_INPUT_TYPE_ARB, GL_POINTS)
        glProgramParameteriARB(self.program_arrow, GL_GEOMETRY_OUTPUT_TYPE_ARB, GL_POINTS) #not sure why this works
        glProgramParameteriARB(self.program_arrow, GL_GEOMETRY_VERTICES_OUT_ARB, 200)


        self.compileProgram( self.program_arrow, vshader, fshader, gshader)
 
        self.arrow_pass = False
           
        




    def init_shaders(self):
        from OpenGL.GL.shaders import *
        #from glutil import compileShader, compileProgram
        vertexsource = """
        //attribute vec3 position; 
        //attribute vec3 normal; 
        //varying vec3 norm; 
        void main() 
        { 
            gl_Position = gl_ModelViewProjectionMatrix * vec4( gl_Vertex.xyz,1);
            //norm = normal;
            gl_TexCoord[0] = gl_MultiTexCoord0;
            gl_FrontColor = gl_Color;
        }
        """ 
        fragmentsource = """
        #version 120
        uniform sampler2D col;      //texture to draw on the sprite
        void main() 
        { 

            vec3 n;
            n.xy = gl_PointCoord.st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
            float mag = dot(n.xy, n.xy);

            if (mag > 1.) discard;   // kill pixels outside circle
            //if (mag > color_life) discard;   // kill pixels outside circle
            //if (mag > 1.0) discard;   // kill pixels outside circle
            
            float snorm = gl_Color.r;
            //float dnorm = clamp(gl_Color.a, 0.0, 1.0);
            float dnorm = gl_Color.a;
            //gl_FragColor = gl_Color * alpha_life;
            vec4 color = vec4(snorm, 0., 1. - snorm, 1.);
            gl_FragColor = color * dnorm;
        }
        """ 

        vshader = compileShader( vertexsource, GL_VERTEX_SHADER )
        fshader = compileShader( fragmentsource, GL_FRAGMENT_SHADER )
        self.program = compileProgram( vshader, fshader )


    def compileProgram(self, program, vertex_shader, fragment_shader, geometry_shader=None):
        from OpenGL.GL import *

        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        if geometry_shader is not None:
            glAttachShader(program, geometry_shader)

        
        glLinkProgram(program);

        glValidateProgram( program )
        validation = glGetProgramiv( program, GL_VALIDATE_STATUS )
        if validation == GL_FALSE:
            raise RuntimeError(
                """Validation failure (%s): %s"""%(
                validation,
                glGetProgramInfoLog( program ),
            ))
        link_status = glGetProgramiv( program, GL_LINK_STATUS )
        if link_status == GL_FALSE:
            raise RuntimeError(
                """Link failure (%s): %s"""%(
                link_status,
                glGetProgramInfoLog( program ),
            ))


        #cleanup
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        if geometry_shader is not None:
            glDeleteShader(geometry_shader)








