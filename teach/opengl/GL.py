#From: http://www.geometrian.com/Tutorials.php
#Import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *

def resize((width, height)):
    #Divide by 0 == bad.
    if height == 0: height=1
    #Set the OpenGL viewing portal to match that
    #of the window's dimensions.
    glViewport(0, 0, width, height)
    #More Initialisation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    #   Set up the Perspective like this.  The 45
    #   is the viewing angle, the 2nd arg. is the
    #   Aspect ratio of the screen, and the 3rd
    #   and 4th arguments describe the near and far
    #   clipping planes, respectively.  These tell
    #   OpenGL where it can draw objects.  Objects
    #   closer than 0.1 or further than 1000.0 OpenGL
    #   unit will not be drawn.  
    gluPerspective(45, 1.0*width/height, 0.1, 1000.0)
    #More Initialisation
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
def init():
    #Enable Masking, Blending, etc.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    #Enable Texturing.  You can use glColor3f() or
    #glColor4f(), but this must be disabled first with
    #glDisable(GL_TEXTURE_2D).  Colors are defined in
    #OpenGL as a tuple of red, green, and blue, (alpha
    #too if using glColor4f).  These values are on a
    #scale of 0.0 to 1.0.  glColor3f(1,1,1) is white,
    #glColor3f(0,0,0) is black.  You can see this here:
    #http://www.pygame.org/project/687/?release_id=1217
    glEnable(GL_TEXTURE_2D)
    #Smooth shading.  This only really does anything
    #when you use Lighting.
    glShadeModel(GL_SMOOTH)
    #When the screen is cleared, this is the background
    #color.  
    glClearColor(0.0, 0.0, 0.0, 0.0)
    #More stuff with clearing.  Leave it alone.
    glClearDepth(1.0)
    #This lets objects be drawn over and under each
    #other, regardless of the order they're drawn in.
    glEnable(GL_DEPTH_TEST)
    #More stuff with transparency.
    glEnable(GL_ALPHA_TEST)
    #More stuff
    glDepthFunc(GL_LEQUAL)
    #Keep the perspective calculations at their best.
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    #More stuff
    glAlphaFunc(GL_NOTEQUAL,0.0)
