

import numpy as np


def prepare_arrays(system):
    tmp = np.ndarray((5,4), dtype=np.float32)
    make_vbo(a)
    #avbo = vbo.VBO(data=a, usage=OpenGL.GL.GL_DYNAMIC_DRAW, target=OpenGL.GL.GL_ARRAY_BUFFER)
    #avbo.bind()
    #vbo = glGenBuffers(1)
    #glBindBuffer(GL_ARRAY_BUFFER, vbo)
    #rawGlBufferData(GL_ARRAY_BUFFER, n_vertices * 2 * 4, None, GL_STATIC_DRAW)

    """
    pos = np.ndarray((clsph.max_num, 4), dtype=np.float32)
    col = np.ndarray((clsph.max_num, 4), dtype=np.float32)

    clsph.pos_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    clsph.pos_vbo.bind()
    clsph.col_vbo = vbo.VBO(data=col, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    clsph.col_vbo.bind()

    clsph.loadData()
    """
    
