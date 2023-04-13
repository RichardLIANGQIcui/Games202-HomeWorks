#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 50
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

//为了实现PCSS，需要定义一些变量
#define FRUSTUM_SIZE  400.0//表示视椎体大小
#define NEAR_PLANE 0.01
#define LIGHT_WORLD_SIZE 5.0//是我们自行设定，可根据效果调节的光源在世界空间的大小
#define LIGHT_SIZE_UV LIGHT_WORLD_SIZE / FRUSTUM_SIZE//光源在ShadowMap上的UV单位大小


uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}
//将一个四维的rgba颜色值转化为一个浮点数灰度值表示深度图
float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];
//计算偏移值
void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
	//要求平均深度，我就先要确定一个采样范围
//这个采样范围是由着色点连接光源的中间，在shadow map上形成的一片区域，shadow map默认在以光源为起点，在近平面的距离上
//这里光源区域在定义时已经计算好，那么根据相似三角形，通过距离的比例就可以求出采样范围大小
  int blockerNum = 0;
  float blockDepth = 0.;//注意vscord对赋值很严格，类型一定要匹配，千万不要漏了小数点，否则结果全黑

  float posZFromLight = vPositionFromLight.z;

  float searchRadius = LIGHT_SIZE_UV * (posZFromLight - NEAR_PLANE) / posZFromLight;

  poissonDiskSamples(uv);
  for(int i = 0; i < NUM_SAMPLES; i++){
    float shadowDepth = unpack(texture2D(shadowMap, uv + poissonDisk[i] * searchRadius));
    if(zReceiver > shadowDepth){
      blockerNum++;
      blockDepth += shadowDepth;
    }
  }

  if(blockerNum == 0)
    return -1.;//注意vscord对赋值很严格，类型一定要匹配，千万不要漏了小数点，否则结果全黑
  else
    return blockDepth / float(blockerNum);

}

float PCF(sampler2D shadowMap, vec4 coords) {
  //整个结果跟采样数量、步长、还有EPS大小有关，但是总体是可以做到对边界的模糊
//这里的采样实际上就是需要计算着色点在shadow map上的偏移值，如果没有使用PCSS给定的范围，就用框架给的采样函数
//这个偏移值 = 采样函数*滤波范围，滤波范围=步长/shadow map的尺寸，在src\engine.js文件中可以找到SM纹理的分辨率
//由于PCF输入的坐标coords归一到了[0,1]的范围
//那么给定采样点的偏移值poissonDiskSamples[i]也需要缩小一定范围以迎合coords坐标的尺寸
//因此需要给定Stride以缩小尺寸
  float shadingDepth = coords.z;
float vb=0.0;
float stride = 10.0;
float shdowmapSize = 2048.0;
float filterRange = stride/shdowmapSize;

poissonDiskSamples(coords.xy);

for(int i=0;i<NUM_SAMPLES;i++)
{
  float sampleDepth = unpack(texture2D(shadowMap,coords.xy+poissonDisk[i]*filterRange));
 // 这里的0.1是浮点数的精度补偿，不加会产生严重的模型上的阴影和噪点
 //该值越大越好，这里取到0.05就看到模型身上有黑色的噪点出现
 //这里可以解释为sampleDepth损失精度较多，需要一个较大数进行补偿，不然下面的等式和容易成立，容易成立就意味着黑点越多
 float res = shadingDepth>sampleDepth+0.05?0.0:1.0;
 vb+=res;
}

return vb/float(NUM_SAMPLES);
}



float PCF2(sampler2D shadowMap, vec4 coords,float filterSize) {

float shadingDepth = coords.z;
float vb=0.0;
float stride = 10.0;  // 步长，压缩尺寸
  float shadowmapSize = 2048.0; // SM分辨率
  float filterRange = stride / shadowmapSize; 

poissonDiskSamples(coords.xy);

for(int i=0;i<NUM_SAMPLES;i++)
{
  float sampleDepth = unpack(texture2D(shadowMap,coords.xy+poissonDisk[i]*filterSize));
  float res = shadingDepth>sampleDepth+EPS?0.0:1.0;
 vb+=res;
}

return vb/float(NUM_SAMPLES);

}


float PCSS(sampler2D shadowMap, vec4 coords){

  float zReceiver = coords.z;
  // STEP 1: avgblocker depth
  float dblocker = findBlocker( shadowMap,  coords.xy, zReceiver);

  if(dblocker < -EPS)
  {
    return 1.0;
  }

  // STEP 2: penumbra size
  float penumbraSize = LIGHT_SIZE_UV*(zReceiver-dblocker)/dblocker;

  // STEP 3: filtering
  float vb = PCF2(shadowMap, coords,penumbraSize);
  return vb;

}


// 使用bias偏移值优化自遮挡
float getBias(float ctrl) {
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float m = 200.0 / 2048.0 / 2.0; // 正交矩阵宽高/shadowmap分辨率/2
  float bias = max(m, m * (1.0 - dot(normal, lightDir))) * ctrl;
  return bias;
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  //shadowCoord这个坐标是着色器插值形成的，所以需要从-1-1转换到0-1之间
  float lightDepth = unpack(texture2D(shadowMap,shadowCoord.xy)); // 将RGBA值转换成[0,1]的float
  float shadingDepth = shadowCoord.z; // 当前着色点的深度

//调整这里的EPS发现对阴影质量影响不大，但是对模型身上噪点的影响很大。
  float bias = getBias(1.4);
  return lightDepth + EPS <= shadingDepth-bias ? 0.0 : 1.0;  
 
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {

    //因为纹理坐标是在[0,1]之间，所以要把裁剪的坐标进行转换
//把裁剪坐标转换为NDC坐标（-1~1）
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  //把[-1,1]的NDC坐标转换为[0,1]的坐标
  shadowCoord.xyz = (shadowCoord.xyz + 1.0) / 2.0;

  float visibility;
  //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  //visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
}