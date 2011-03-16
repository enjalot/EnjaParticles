void GE_addRect(int num, float4 min, float4 max, float spacing, float scale, std::vector<float4>& output);
void addRect(int num, float4 min, float4 max, float spacing, float scale, std::vector<float4>& output);
void addSphere(int num, float4 center, float radius, float spacing, float scale, std::vector<float4>& output);
void addCircle(int num, float4 center, float radius, float spacing, float scale, std::vector<float4>& output);
void addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing, std::vector<float4>& output);
void addRandRect(int num, float4 min, float4 max, float spacing, float scale, float4 dmin, float4 dmax, std::vector<float4>& output);
void addRandArrangement(int num, float scale, float4 dmin, float4 dmax, std::vector<float4>& output);
void addRandSphere(int num, float4 center, float radius, float spacing, float scale, float4 dmin, float4 dmax, std::vector<float4>& output);
