


#VBOS
global position, normal 
vert_vbo = vbo.VBO(data=points, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
normal_vbo = vbo.VBO(data=normals, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
tri_indices = vbo.VBO(data=tris, usage=GL_DYNAMIC_DRAW, target=GL_ELEMENT_ARRAY_BUFFER)


def draw_triangles(vbos):

    glColor3f(.5,.5,.5)
    scale = .1
    glScalef(scale, scale, scale)

    #glVertexAttribPointer( normal, 3, GL_FLOAT, 0, 12, normal_vbo )
    #glVertexAttribPointer( position, 3, GL_FLOAT, 0, 12, vert_vbo ) 

    if vbos:
        vert_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, vert_vbo)

        normal_vbo.bind()
        glNormalPointer(GL_FLOAT, 0, normal_vbo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        tri_indices.bind()
        glDrawElements( GL_TRIANGLES, len(tris)*3, GL_UNSIGNED_INT, tri_indices )

        """
        glColor3f(1, 0, 0)
        glPointSize(1)
        vert_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, vert_vbo)
        glDrawArrays(GL_POINTS, 0, len(points))
        """



    else:
        glBegin(GL_TRIANGLES)
        #glBegin(GL_POINTS)
        for i,t in enumerate(tris):
            if i > 10000:
                glEnd()
                return

            v1 = points[t[0]]
            v2 = points[t[1]]
            v3 = points[t[2]]
            n1 = normals[t[0]]
            n2 = normals[t[1]]
            n3 = normals[t[2]]

            glVertex3f(v1[0], v1[1], v1[2])
            #glNormal3f(n1[0], n1[1], n1[2])
            glVertex3f(v2[0], v2[1], v2[2])
            #glNormal3f(n2[0], n2[1], n2[2])
            glVertex3f(v3[0], v3[1], v3[2])
            glNormal3f(n3[0], n3[1], n3[2])

        glEnd()

        #print "verts: ", v1, v2, v3
        #print "normals: ",n1, n2, n3


