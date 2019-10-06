#include <iostream>
#include <testlib/testlib_test.h>
#ifdef _MSC_VER
#  include <vcl_msvc_warnings.h>
#endif
// The following test is to determine if there are platform differences in
// roc calculations. The reference counts are obtained on
// a Dell XPS M1710 Centro Duo running Windows XP
// compiled with Visual Studio 2005 version 8.0.50727.42
//
static void test_platform_computations()
{
  double rnums[100] = {0.8074965495261911,0.5560984670738593,0.6520680644460387,0.06535145828096316,0.03646050891592551,0.8856246747465122,0.3195027341645003,0.49350526157777075,0.9078552016131822,0.6488073903177274,0.408494573349602,0.48716413835868044,0.6388263848947237,0.08871511530311905,0.8537378265157126,0.7827477068519928,0.05202257380699154,0.40852280386954143,0.005177833030198396,0.8556002063125417,0.10663867673893332,0.12548951606722542,0.9940366746525422,0.6303539595703125,0.29914212721274225,0.5693910489933661,0.34196861020650343,0.5650025012893494,0.26268161829681674,0.6837663742468539,0.022465876042003164,0.07149723971157855,0.3548264166836345,0.0349589839291265,0.6139713026924012,0.5843331013528981,0.7160000317889109,0.9462438686260074,0.7602334761766886,0.8015853945009053,0.6639774579819193,0.5377210647564661,0.7550556431464901,0.9459851881883636,0.5573387812429861,0.4122315486892406,0.761018968493948,0.31563122861805115,0.2581966540302438,0.8428404996958745,0.4190503582874445,0.7506287273287019,0.995515035733427,0.1590741254490206,0.39658448224544135,0.6791314876171233,0.6406886190497926,0.12411514151989411,0.7826131795530401,0.0947983862642252,0.9246885872608817,0.17787127289388666,0.022379703376351624,0.2932129917633199,0.2607111292789623,0.6401502081374206,0.2673240602298615,0.34722780357495625,0.7033723480359763,0.22791865944818002,0.5063050917359135,0.031596574956905085,0.4451756940057324,0.3850781597523055,0.08725473344846903,0.28096784762820326,0.4496606582723054,0.2260040343032849,0.6906702512030277,0.6018363600110799,0.8089720392225128,0.10188889278339078,0.9080570716499875,0.5070379737468548,0.8842834519616312,0.9240176198895041,0.8856773682736359,0.21382498198353486,0.6235723226826688,0.28386741175208346,0.6183533080437744,0.8665971784085786,0.9201999746466926,0.05594875230390344,0.11204821630786088,0.8350006034516735,0.47502428064096014,0.670870592551598,0.024793482859391855,0.5540327558234703};

  unsigned counts[100] = {100,99,99,96,93,93,91,90,89,87,86,84,83,81,81,81,80,80,79,79,79,79,78,76,76,76,75,72,72,70,68,68,66,66,66,64,63,63,63,62,61,59,57,57,57,55,55,55,54,53,52,50,50,50,49,49,46,44,44,43,43,42,40,39,37,34,33,32,30,29,28,27,26,26,26,26,24,22,22,20,20,17,17,17,16,15,13,12,12,9,9,7,7,4,4,2,2,2,2,2};

// double test
  bool faild = false;
  std::cout << "Threshold test for double\n";
  double t = 0.0;
  for (unsigned  k=0; k<100; t+=0.01, k++)
  {
    unsigned c = 0;
    for (double rnum : rnums)
      if (rnum>t)
        c++;
#if 0
    std::cout << c << ' ';
#endif
    if (c!=counts[k]){
      std::cout << "count[" << k << "]= " << c << " should be "
               << counts[k] << '\n';
      faild = true;
    }
  }
  std::cout << "\n\n";
  TEST("double threshold", faild, false);
  bool failf = false;
  std::cout << "Threshold test for float\n";
  float tf = 0.0f;
  for (unsigned k=0; k<100; tf+=0.01f, k++)
  {
    unsigned c = 0;
    for  (double rnum : rnums)
      if (static_cast<float>(rnum)>tf)
        c++;
#if 0
    std::cout << c << ' ';
#endif
    if (c!=counts[k]){
      std::cout << "count[" << k << "]= " << c << " should be "
               << counts[k] << '\n';
      failf = true;
    }
  }
  std::cout << "\n\n";
  TEST("float threshold", failf, false);
}

TESTMAIN( test_platform_computations );
