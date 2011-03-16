/** SWIG declarations for float4 type declared in structs.h */
typedef struct float4
{
    float x;
    float y;
    float z;
    float w;

    float4();
    float4(float xx, float yy, float zz, float ww);
	void set(float xx, float yy, float zz, float ww=1.);
    float length(); 
    /* NOTE: we dont expose the overloaded operators because we need to
     * "%extend" the class with them below. SWIG cant support friend and
     * operator at the same time */
} float4;

/* Add support for operator overloads */
%extend float4
{
    char *__str__() {
        static char tmp[1024];
        sprintf(tmp,"float4(%g,%g,%g,%g)", $self->x,$self->y,$self->z,$self->w);
        return tmp;
    }

    float4 __add__(float4 *other) {
        float4 v;
        v.x = $self->x + other->x;
        v.y = $self->y + other->y;
        v.z = $self->z + other->z;
        v.w = $self->w + other->w;
        return v; 
    }

    float4 __sub__(float4 *other) {
        float4 v;
        v.x = $self->x - other->x;
        v.y = $self->y - other->y;
        v.z = $self->z - other->z;
        v.w = $self->w - other->w;
        return v; 
    }

    float4 __rmul__(float r) {
        float4 v; 
        float d = r; 
        v.x = $self->x * d; 
        v.y = $self->y * d; 
        v.z = $self->z * d; 
        v.w = $self->w * d; 
        return v;
    }

    float4 __mul__(float r) {
        float4 v; 
        float d = r; 
        v.x = $self->x * d; 
        v.y = $self->y * d; 
        v.z = $self->z * d; 
        v.w = $self->w * d; 
        return v;
    }
    
    float4 __div__(float r) {
        float4 v; 
        float d = 1./r; 
        v.x = $self->x * d; 
        v.y = $self->y * d; 
        v.z = $self->z * d; 
        v.w = $self->w * d; 
        return v;
    }
}

