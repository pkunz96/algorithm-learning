import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 200, 1)
heap_sort_sgd_y = [3.3252820969, 3.219183892, 3.156235069, 3.1158191562, 3.0880790353, 3.0678922832, 3.0525079668, 3.040299505, 3.0303300917, 3.0219037533, 3.0146192312, 3.0082180202, 3.0025622845, 2.997407347, 2.9927653372, 2.9884912968, 2.9845853448, 2.9810039103, 2.97759372, 2.9743762016, 2.9713695943, 2.9684309065, 2.9654837549, 2.962608695, 2.9598839283, 2.9571529031, 2.9543821514, 2.9516392946, 2.9488606155, 2.9460571706, 2.9431609809, 2.9401545525, 2.9370824993, 2.9338839948, 2.930706799, 2.9273991883, 2.9239883423, 2.9204402268, 2.9167916477, 2.9129447937, 2.9090165496, 2.9049050808, 2.900620997, 2.89623487, 2.8917823136, 2.8872112632, 2.8825481236, 2.8778327405, 2.8729992807, 2.8681426942, 2.8632307053, 2.858347863, 2.8534537852, 2.8485661149, 2.8437463641, 2.8389493823, 2.8342311382, 2.8295553178, 2.8249119073, 2.8203737736, 2.815865621, 2.8114600033, 2.8070760518, 2.8028089106, 2.7985541075, 2.7944001406, 2.7902640253, 2.786215812, 2.7822186798, 2.7783648074, 2.7745452821, 2.7707731426, 2.7670733035, 2.7634253204, 2.759800002, 2.756266281, 2.7527869195, 2.7492623478, 2.7459645271, 2.7426507473, 2.7393319756, 2.7361204028, 2.7329411507, 2.729866758, 2.7267606556, 2.7237227112, 2.7206874192, 2.7177187651, 2.7148221433, 2.7119041234, 2.709026143, 2.7061412036, 2.7033245564, 2.700466916, 2.6976510286, 2.6947093457, 2.6918614358, 2.689171344, 2.6863271445, 2.6835125387, 2.6807445139, 2.6779536605, 2.6751724109, 2.6724545211, 2.6697880477, 2.6669129506, 2.664173007, 2.6612927914, 2.6585968286, 2.6557186246, 2.6528697163, 2.6500793844, 2.6471660584, 2.644252263, 2.6413734704, 2.6384395286, 2.635525465, 2.6324829608, 2.6295287088, 2.6264481172, 2.6233534068, 2.6202828959, 2.6171083301, 2.6139221638, 2.6108102426, 2.6075692996, 2.604289107, 2.6009779423, 2.5975910574, 2.5941944942, 2.5907337293, 2.587233372, 2.5836683288, 2.580212079, 2.5765978247, 2.5729154274, 2.5692902058, 2.5657449216, 2.5620001927, 2.5582693443, 2.5546117201, 2.5507885367, 2.5470223501, 2.5432395339, 2.5393959954, 2.5355161875, 2.5316229612, 2.5277353674, 2.5237483978, 2.5197706297, 2.515863359, 2.5118579119, 2.5078560114, 2.5037492514, 2.4997169301, 2.4957081899, 2.49150718, 2.4874904305, 2.483353287, 2.4792027101, 2.4750394225, 2.4708479196, 2.4666918963, 2.4623879194, 2.4583241642, 2.4540107921, 2.4496831149, 2.4455412775, 2.4412307441, 2.4368741289, 2.4326308742, 2.4283385277, 2.4240199849, 2.4196043238, 2.4151877016, 2.4108091146, 2.4063914567, 2.4019337445, 2.3974706233, 2.3930359781, 2.3885234296, 2.3839977309, 2.3795321509, 2.3749381229, 2.370374836, 2.3658699244, 2.3611866757, 2.3565939292, 2.3519627452, 2.3472737148, 2.3426741287, 2.3379484564, 2.3332533836, 2.3286257833, 2.3238699958, 2.3191962242, 2.3144625276, 2.3097554743, 2.3049882278, 2.3003690094]

heap_sort_adam_y = [2.8299081028, 2.1271393299, 1.5221110806, 1.0777477846, 0.7578548789, 0.5793738365, 0.4933993369, 0.4408500008, 0.4086881541, 0.3961523585, 0.3824680857, 0.3659622222, 0.3474688195, 0.32349113, 0.3137637116, 0.3033365346, 0.2995770257, 0.2919999063, 0.2919967361, 0.2790586147, 0.2696980946, 0.2691042814, 0.2686550934, 0.2569160573, 0.2574819736, 0.2576934397, 0.2518563215, 0.2431813162, 0.242945781, 0.2295658998, 0.22916485, 0.2242318653, 0.2316986974, 0.2233080491, 0.2227115538, 0.2134706024, 0.2156702131, 0.2094105519, 0.2059709113, 0.1978281364, 0.1982292188, 0.2003298514, 0.1919073202, 0.1918195086, 0.192081077, 0.1921191681, 0.1868574675, 0.1817979552, 0.1823541662, 0.1920540622, 0.1814479223, 0.1854793495, 0.1789405402, 0.1808962785, 0.175718125, 0.1730406657, 0.1811633762, 0.1801770842, 0.1765008159, 0.1689720917, 0.1709957039, 0.1596465884, 0.1668739598, 0.1614521472, 0.1599526806, 0.1666014725, 0.1762472205, 0.1627920857, 0.1597192548, 0.1558164656, 0.1558673112, 0.1574271396, 0.1640911652, 0.1549310694, 0.1454087924, 0.1448570695, 0.1415992454, 0.1434517289, 0.1508363504, 0.1481505809, 0.1415902795, 0.1472312249, 0.1514394982, 0.1530910786, 0.1429471914, 0.1506974883, 0.1446538977, 0.1458839448, 0.1494289842, 0.1451133518, 0.1349545056, 0.1332050879, 0.1300496301, 0.1286333185, 0.1399049358, 0.1381858969, 0.1359297074, 0.1330879284, 0.1325853737, 0.1293563861, 0.1349469535, 0.1379769407, 0.1294994894, 0.1282891352, 0.1243453128, 0.1253568279, 0.1235208204, 0.1259131869, 0.1257451139, 0.1265816251, 0.1219698302, 0.1169688422, 0.1211785628, 0.1197968414, 0.1259366274, 0.1182066593, 0.1217299495, 0.1248244997, 0.1203712225, 0.1228968631, 0.1225730823, 0.1220436925, 0.1170808645, 0.1117839525, 0.1056824811, 0.1155517972, 0.114211069, 0.1110707726, 0.1097892271, 0.1093532695, 0.1184063088, 0.1157155382, 0.1198478099, 0.1225828221, 0.1190987788, 0.113658756, 0.1134263519, 0.1117986832, 0.109546402, 0.1070630704, 0.1141184904, 0.1076843417, 0.1115545109, 0.1029843488, 0.1029689359, 0.1086527063, 0.1075136019, 0.114044141, 0.1172596095, 0.1113791927, 0.101486587, 0.100799636, 0.0979703851, 0.0963608408, 0.0989791774, 0.1002665837, 0.0995018939, 0.0970552829, 0.1022303277, 0.1102948394, 0.1097497321, 0.1033664341, 0.101108992, 0.1089289417, 0.1021538898, 0.0984426178, 0.101206325, 0.1001096438, 0.1097439351, 0.1061303429, 0.1092383591, 0.1013788804, 0.1019639764, 0.1014070129, 0.1031203484, 0.099826464, 0.1002433496, 0.0968537098, 0.0977723268, 0.1024764464, 0.102233151, 0.098536416, 0.0983064077, 0.0925145247, 0.0931279114, 0.0879273866, 0.0914026494, 0.0915110065, 0.0985867041, 0.096009999, 0.0913320873, 0.0911698379, 0.0931157568, 0.0889180927, 0.0886658682, 0.0960077923, 0.0983462459, 0.0917005925, 0.0895411009, 0.0873100753]

insertion_sort_adam_y = [0.8298394084, 0.6108403541, 0.4913505018, 0.4430467114, 0.3659720998, 0.3277196027, 0.3038911968, 0.2762056161, 0.2594371121, 0.2443124391, 0.2267865911, 0.2157302778, 0.2062190361, 0.2021952011, 0.1951139662, 0.1859651264, 0.1810466833, 0.1752687106, 0.1773687238, 0.1780210771, 0.1972173918, 0.1800964624, 0.1679360997, 0.1643677792, 0.1602454092, 0.1622438077, 0.1644971073, 0.1612311807, 0.1611746978, 0.1528352657, 0.1545188138, 0.1562053869, 0.1713278107, 0.1603003358, 0.1532667251, 0.1448347345, 0.1443875032, 0.1545570921, 0.1475297939, 0.1493541179, 0.1431275243, 0.1366992695, 0.1356050363, 0.1429438218, 0.1477409722, 0.14270774, 0.1357397055, 0.1335554579, 0.1397610493, 0.1429087566, 0.1372896833, 0.1371905291, 0.1291220523, 0.1272035753, 0.1333802463, 0.1322889375, 0.1358103165, 0.1314281859, 0.1279843841, 0.1314625209, 0.140830243, 0.1428766819, 0.1302544475, 0.1264033839, 0.1230533309, 0.1271972787, 0.1259898283, 0.123762941, 0.1301711937, 0.1331473617, 0.1342999544, 0.1272778697, 0.1232282147, 0.1259628469, 0.1268796884, 0.1309823077, 0.128173993, 0.1216429723, 0.1164385397, 0.1131066782, 0.1148079988, 0.1226332923, 0.123998086, 0.1210453948, 0.1248673387, 0.1244403189, 0.1329319756, 0.1279053967, 0.1285845609, 0.1190126436, 0.1146901743, 0.1179536683, 0.123632092, 0.120571807, 0.1205479857, 0.1215588301, 0.1147756614, 0.1216678657, 0.1131905178, 0.1126401629, 0.1150881927, 0.1100409487, 0.1134639671, 0.1305224327, 0.1136769848, 0.1106456723, 0.1101525482, 0.1114952583, 0.1287451228, 0.1216591876, 0.1173238084, 0.1207375741, 0.1183423372, 0.1075534699, 0.1087582838, 0.1172779324, 0.1169362869, 0.1120299269, 0.1206091046, 0.1100030681, 0.1069007181, 0.1055292105, 0.1088507771, 0.1096398234, 0.1155818868, 0.1117782732, 0.1083947513, 0.1024399735, 0.1067184145, 0.1089202445, 0.1051638704, 0.1038492788, 0.1108559044, 0.1188384881, 0.1233080225, 0.1063762791, 0.106638479, 0.104740628, 0.1034509721, 0.1039131293, 0.1067695739, 0.1040754113, 0.1050708294, 0.1114251148, 0.1106153354, 0.1101886639, 0.1020548865, 0.1002335781, 0.0987511734, 0.1031425465, 0.100988402, 0.1038294891, 0.1141019491, 0.1036131713, 0.0990857482, 0.0956687205, 0.0992275402, 0.1029056725, 0.1124063465, 0.1085294066, 0.1074157367, 0.1083922526, 0.1024571452, 0.099670412, 0.1081949598, 0.1180285942, 0.1070614802, 0.1110604745, 0.102724947, 0.1035454832, 0.1023593647, 0.1043316703, 0.1007040469, 0.0991204041, 0.1168925157, 0.1079613883, 0.1122260499, 0.1110477028, 0.1041228315, 0.1078512277, 0.1006218614, 0.1050830623, 0.0956109511, 0.0932336231, 0.0927739884, 0.0943218381, 0.0998561447, 0.1132671167, 0.1086591072, 0.1138723474, 0.1132046822, 0.1030108128, 0.0983719896, 0.1025214498, 0.0945536559, 0.0952645848, 0.0995306149, 0.0930150314, 0.1007367391, 0.1103423145]

insertion_sort_sgd_y = [1.0987907052, 0.9026573151, 0.8637669459, 0.8465039358, 0.837207526, 0.8313943818, 0.8272123188, 0.8238766342, 0.8210916296, 0.8186172023, 0.8163867816, 0.8143225387, 0.8124375194, 0.8105582595, 0.8087002262, 0.8069119528, 0.8050842658, 0.8033235744, 0.801557906, 0.7997948825, 0.7979454696, 0.7960922122, 0.7942182198, 0.7923166454, 0.7904384658, 0.7885759696, 0.7865774184, 0.7847340554, 0.7829087675, 0.7811132967, 0.7793234959, 0.7775787488, 0.7757605389, 0.7740555778, 0.7723643258, 0.7708015069, 0.7690626904, 0.7674849592, 0.7657193318, 0.7640332542, 0.7624424659, 0.7607367709, 0.7591072731, 0.7573516406, 0.7557250597, 0.7539395653, 0.7522835657, 0.7505436093, 0.7488196567, 0.7469932772, 0.7452824935, 0.7435047477, 0.7415972203, 0.7398487478, 0.7380106151, 0.7361288592, 0.7342635766, 0.7323812768, 0.7305024341, 0.7286869735, 0.7269467674, 0.7250816151, 0.7233138084, 0.7215080895, 0.7198335603, 0.7181795388, 0.7164842561, 0.7148115002, 0.7132091597, 0.7115919851, 0.7098735422, 0.7082242519, 0.70663248, 0.7049996145, 0.7033657506, 0.7016399503, 0.6999930628, 0.6983418427, 0.6966335066, 0.6948884763, 0.6931832097, 0.6913720295, 0.6896202452, 0.6878393516, 0.6858987771, 0.6840452701, 0.6821924821, 0.6802651286, 0.6783791892, 0.6763348728, 0.6744547114, 0.6724785976, 0.6704836898, 0.6684961356, 0.6665376909, 0.6645465158, 0.6626035906, 0.6606290489, 0.6587942205, 0.656878449, 0.6549787112, 0.653132502, 0.6512530297, 0.6495031975, 0.6478093453, 0.6461043656, 0.6444573663, 0.6427891627, 0.6412359029, 0.6396862119, 0.6382195614, 0.6368542537, 0.6354720108, 0.6341190971, 0.6329351403, 0.6317863539, 0.6305949353, 0.6295427047, 0.6284911819, 0.6274517141, 0.6265108138, 0.6255888268, 0.6246780455, 0.6238590479, 0.6231045537, 0.6222206093, 0.6214392334, 0.6207157671, 0.620010104, 0.6192802899, 0.6186236478, 0.618019022, 0.6173090711, 0.6166817434, 0.6161041223, 0.6155395918, 0.6149066351, 0.6143716052, 0.6137833335, 0.6132751368, 0.6126367599, 0.6121395193, 0.6116274633, 0.6111275256, 0.6106258519, 0.6100826748, 0.6095534526, 0.6090745926, 0.6085861027, 0.6080647409, 0.6075454764, 0.6071428694, 0.6066121906, 0.6060715243, 0.6056330241, 0.6051174626, 0.6045769565, 0.604022529, 0.6036377661, 0.603029035, 0.6025101691, 0.6020668708, 0.6015420072, 0.6010558642, 0.6006171703, 0.6000205576, 0.5994982086, 0.5989009701, 0.5984149724, 0.597804781, 0.5973473787, 0.5967299044, 0.5962056667, 0.5956120566, 0.5951059088, 0.594438605, 0.5939335823, 0.5933879726, 0.5927302912, 0.5921617039, 0.5915508606, 0.5909201764, 0.5903555602, 0.5897596143, 0.5891590081, 0.5883637294, 0.5877442881, 0.5870739035, 0.586507082, 0.5857388005, 0.5850781091, 0.5844065323, 0.583625447, 0.5830133893, 0.5821281634, 0.5815500766, 0.5808496885, 0.5800355263, 0.5791974701, 0.5784066878]

quick_sort_adam_y = [1.6396272033, 1.1149452776, 0.8162890375, 0.6510547176, 0.5420911461, 0.4979615994, 0.473206792, 0.4557594657, 0.4339378811, 0.4221706167, 0.4030967169, 0.3863744698, 0.3786571138, 0.3813740872, 0.3723869435, 0.3616236076, 0.3481989354, 0.333428774, 0.3244756982, 0.3216350116, 0.3190911934, 0.3158731163, 0.307266701, 0.3029968515, 0.3018147349, 0.2910684813, 0.3257780634, 0.3079282604, 0.3082333133, 0.2978112958, 0.2949422337, 0.2838231344, 0.2851630766, 0.2813282628, 0.2638333142, 0.2815278117, 0.2730709892, 0.2693395317, 0.2728276588, 0.2663465347, 0.2532621566, 0.2632659357, 0.2608965859, 0.2579153646, 0.2517645657, 0.2660884727, 0.2565223817, 0.2480209265, 0.2307607159, 0.2419758085, 0.2464355379, 0.2523451336, 0.2421690747, 0.269351624, 0.261068603, 0.2542865574, 0.2481016479, 0.2441630326, 0.2298729774, 0.2304778285, 0.2289479338, 0.2525284328, 0.2497021463, 0.2270143442, 0.2137612198, 0.212275872, 0.2281452175, 0.2268561218, 0.2244109288, 0.2177877985, 0.2152611781, 0.2205729615, 0.2144297101, 0.2110479549, 0.2240103316, 0.2295431774, 0.2360527273, 0.2423256617, 0.2273669615, 0.2297612131, 0.2191704549, 0.2219066098, 0.2142538279, 0.2131333034, 0.2115537375, 0.211312782, 0.2054037768, 0.2188817859, 0.2039003558, 0.2210204378, 0.2102653105, 0.1963856667, 0.1902201064, 0.1891183201, 0.1986793224, 0.2027146704, 0.2274898496, 0.2074954342, 0.1916001886, 0.1899221726, 0.2048472688, 0.2144095916, 0.2142063268, 0.2077537235, 0.1978241652, 0.2028562203, 0.1884083133, 0.1999693755, 0.1884861197, 0.2080553044, 0.1917763315, 0.1834453829, 0.1885284875, 0.1848307047, 0.1959839892, 0.1981381029, 0.1833846029, 0.1849662513, 0.190381363, 0.1844851561, 0.1974624358, 0.2008245718, 0.1919379197, 0.1898435093, 0.1847378612, 0.1814138554, 0.1888157874, 0.1800642516, 0.1752939206, 0.1936333086, 0.1942982376, 0.1977939084, 0.1873053946, 0.1926537342, 0.1808671933, 0.1661706138, 0.1864909735, 0.1809405386, 0.1954225115, 0.203458311, 0.1814704128, 0.1858883146, 0.1797226388, 0.2028044034, 0.1912658736, 0.1806080081, 0.17538069, 0.1813245546, 0.1894873213, 0.2006164677, 0.2027944662, 0.1724275723, 0.1624191962, 0.1584460493, 0.1769956443, 0.1796109863, 0.1864306498, 0.1908718031, 0.1614407375, 0.1701543946, 0.1947136708, 0.1855922956, 0.1826175824, 0.1815942973, 0.1788663864, 0.1643418018, 0.1703744773, 0.1739160698, 0.1743896231, 0.1691019405, 0.1587484237, 0.1754713096, 0.1865988113, 0.1713786665, 0.1732795481, 0.1709512658, 0.1605135947, 0.1695548464, 0.1694632154, 0.180737352, 0.1699686013, 0.1678086985, 0.1667762799, 0.165122373, 0.1630477682, 0.1698108148, 0.1679104697, 0.1813258268, 0.179183051, 0.1795284711, 0.171729032, 0.1698757503, 0.1637053788, 0.1540427823, 0.155701912, 0.1803893503, 0.1655057054, 0.170202231, 0.1546737673, 0.1627781661]

quick_sort_sgd_y = [2.0547337383, 1.8809311241, 1.8269349635, 1.8017537743, 1.7876436412, 1.7781511694, 1.7709130347, 1.764670983, 1.759092316, 1.7535104305, 1.748048231, 1.7425004989, 1.7368062437, 1.7308936268, 1.7247018665, 1.7182872146, 1.7116546333, 1.7047367394, 1.6976766139, 1.6906149983, 1.6834254265, 1.6762821525, 1.6691258252, 1.6622025967, 1.6553340852, 1.6485946551, 1.6421708539, 1.6355321631, 1.6291491464, 1.6229727566, 1.6169443801, 1.6110430807, 1.6052032933, 1.5995347723, 1.5939688757, 1.5883957446, 1.5829557776, 1.5775852874, 1.5722243339, 1.5668017343, 1.5612312183, 1.5557366759, 1.550185442, 1.5445557311, 1.5387038291, 1.5329388455, 1.52698607, 1.5210431367, 1.5149136633, 1.5086792335, 1.5024602115, 1.4960598573, 1.4896750525, 1.4831504524, 1.4766232893, 1.4699843153, 1.4633765668, 1.4566963911, 1.4499731958, 1.4432846382, 1.4366201609, 1.429879874, 1.4231878743, 1.4165624231, 1.41014231, 1.4039204195, 1.3976439163, 1.3915516883, 1.3856794685, 1.3797904402, 1.3740697354, 1.3685118854, 1.3630207479, 1.3577217534, 1.3524669036, 1.3474337757, 1.342404224, 1.3375715017, 1.3327073529, 1.3280126229, 1.3234940171, 1.3189634979, 1.3145593703, 1.3103185147, 1.3060137853, 1.3018681556, 1.2978777438, 1.293873705, 1.2900131941, 1.2860860601, 1.2823470682, 1.2785135135, 1.2747862488, 1.2712528929, 1.2677434161, 1.2642456815, 1.2608833835, 1.2575172484, 1.2541301623, 1.2509449124, 1.2476873398, 1.2446517646, 1.2415948436, 1.2385039702, 1.235549666, 1.2326435149, 1.2298554927, 1.2269464582, 1.2242612317, 1.2215819731, 1.2188852802, 1.2162964419, 1.2137465626, 1.2112336084, 1.2086584866, 1.2062510699, 1.2038115561, 1.2014639601, 1.199052114, 1.1966694817, 1.1942765787, 1.1920843273, 1.1898469999, 1.1876624152, 1.1854967102, 1.1832190417, 1.1810658686, 1.1788800694, 1.1768389195, 1.17469972, 1.1725681871, 1.1704294048, 1.1684292108, 1.1662745774, 1.1643088534, 1.1622424349, 1.1601595767, 1.1581268162, 1.1560388133, 1.153892383, 1.1519137099, 1.1498430409, 1.1478369832, 1.14567063, 1.1436581239, 1.1414451897, 1.1394604854, 1.1372864395, 1.1353776194, 1.1331247762, 1.1309217177, 1.1286522113, 1.1264254674, 1.1241879798, 1.1219615936, 1.1197011024, 1.1174145676, 1.1151143834, 1.1126831844, 1.1104135998, 1.1080152206, 1.105584152, 1.1031352021, 1.1006163917, 1.0980912559, 1.0954896249, 1.0929953903, 1.0902587473, 1.0877230652, 1.084940955, 1.0822568126, 1.0794572756, 1.0765600167, 1.0737684071, 1.0709695518, 1.0678796843, 1.0650539324, 1.0620013513, 1.0589475222, 1.055950135, 1.0528177656, 1.0497136265, 1.0465631336, 1.0433118716, 1.0402036346, 1.0368822962, 1.0335155651, 1.0303467996, 1.0270876102, 1.0237791762, 1.0205073357, 1.0172052197, 1.0138516091, 1.0105514638, 1.0071943514, 1.0038208403, 1.000419192, 0.9971059114, 0.9939194918, 0.9905387312]

top_down_merge_sort_adam_y = [2.6813815981,2.1052359939,1.6306353137,1.2626612931,0.9921899661,0.8394922465,0.7569566071,0.7133608609,0.6934349239,0.6836123616,0.6719645485,0.6634962559,0.6592397913,0.6617221162,0.6563269272,0.6363293529,0.6344577372,0.6272825375,0.6242816299,0.6209537238,0.622220248,0.6111803874,0.6019767299,0.6014767364,0.5996120498,0.6071047932,0.598465234,0.5922321305,0.5880812556,0.579234831,0.5790435709,0.5772120729,0.5737658925,0.5732841715,0.5614487603,0.5648042224,0.5565027744,0.5541018173,0.5567967258,0.5518064089,0.5512384698,0.5420813523,0.5440264717,0.5395258665,0.5308827087,0.5367819667,0.5443063155,0.5333859846,0.5311721861,0.5180883855,0.5206778422,0.5243736282,0.5226386487,0.5095051378,0.5113941021,0.5040122233,0.4988429621,0.5036217235,0.4941510595,0.5040569194,0.4996298738,0.4959751368,0.5008021519,0.4963801093,0.4951826781,0.4879310541,0.4795862809,0.4806616716,0.4780769572,0.4696952216,0.4752154313,0.4731551781,0.46651683,0.4605542012,0.4716120698,0.4741310328,0.4719452299,0.4644015729,0.4588284455,0.4503278323,0.4507631548,0.4495606124,0.4496402107,0.4420516342,0.4480214417,0.4518543594,0.445854228,0.4429690093,0.4395326711,0.4436841272,0.4380982332,0.4364141822,0.4308439493,0.4393295348,0.4267575704,0.4286839254,0.4246226251,0.4200344235,0.4175991938,0.4105343409,0.4083665218,0.4057090506,0.4112225901,0.4095728826,0.4120262042,0.4108665027,0.4072578512,0.4041788429,0.4036573768,0.4017663375,0.4037157632,0.394352112,0.3961643446,0.3928127307,0.3910591919,0.3879275452,0.3883130103,0.3809211701,0.3848092388,0.376904944,0.3820551932,0.3772432096,0.3780215271,0.3782269061,0.371386217,0.3690881301,0.3711378742,0.3715445213,0.3771632146,0.3745265007,0.3735641036,0.3649561573,0.3688419014,0.3692227062,0.3607168756,0.3580814488,0.3574513849,0.3514186833,0.351041764,0.347273076,0.344807623,0.3477155045,0.3390373401,0.3393238541,0.3384262044,0.337109061,0.3315045163,0.339415079,0.3362858705,0.3318915423,0.3256294169,0.3319536895,0.337258352,0.3350963201,0.3378594313,0.3373142816,0.3350991141,0.3306521159,0.3272902705,0.3202415593,0.3145784438,0.31759285,0.316778373,0.321219625,0.3210140858,0.3131369296,0.3189429082,0.3191147465,0.3141119499,0.3157688249,0.3099840134,0.3068578616,0.3069873005,0.3076813091,0.3079976514,0.3017784301,0.3026028723,0.3033463322,0.3013624214,0.2964122538,0.2903588079,0.3018026371,0.300530253,0.2945265658,0.2948990706,0.2948290035,0.2999103516,0.2995172273,0.2944431603,0.2977640536,0.2989245243,0.2911732364,0.2965340093,0.2860729937,0.2863868475,0.2856190093,0.2748107277,0.272984419,0.2759995628,0.2758887149]

top_down_merge_sort_sgd_y = [3.1979400218, 3.0301893353, 2.9339845777, 2.8770668209, 2.8420978785, 2.8191260397, 2.8029825687, 2.7912186682, 2.782271564, 2.7751012444, 2.7691875994, 2.7642036378, 2.7597354054, 2.7555824518, 2.7517196536, 2.748064965, 2.7445715964, 2.7411125898, 2.7377375364, 2.7343387604, 2.7309072912, 2.7273889482, 2.723847419, 2.7202097774, 2.7165094316, 2.7126664221, 2.7087801099, 2.7047901154, 2.7006477118, 2.6964779794, 2.6920886934, 2.6877771616, 2.6833300591, 2.6788303256, 2.6742454171, 2.6697040796, 2.6651243567, 2.6605669856, 2.6559419334, 2.6514437646, 2.6469055414, 2.6424673945, 2.6381080002, 2.6338253766, 2.6296011657, 2.6254924238, 2.6214093119, 2.6174290627, 2.613518551, 2.6097475588, 2.6060342491, 2.6024748385, 2.5989456475, 2.5954979807, 2.5922419131, 2.5889803469, 2.5857940614, 2.5826687515, 2.5796740055, 2.5766731352, 2.5736628771, 2.5707973689, 2.5679475963, 2.5652212203, 2.5624158531, 2.5597703457, 2.5570338219, 2.5543759465, 2.5517585576, 2.5492055267, 2.5465595126, 2.544026792, 2.5415081829, 2.5389403999, 2.5363954753, 2.5339074135, 2.5314395428, 2.5289241076, 2.5263527185, 2.5239037871, 2.5213748217, 2.5188723952, 2.5163707733, 2.5138249993, 2.5113454312, 2.508870557, 2.5063026249, 2.50376302, 2.5012137443, 2.4987276644, 2.4959859401, 2.4934398979, 2.4907794595, 2.4880770743, 2.4853632599, 2.4826670587, 2.479857102, 2.4770301878, 2.474244684, 2.4713759273, 2.4684538841, 2.4654366821, 2.4624522254, 2.4594261944, 2.4563713744, 2.4531646296, 2.4500580281, 2.4468796253, 2.4434349835, 2.440139398, 2.4367966875, 2.4332205951, 2.4297084734, 2.4260248542, 2.4225265607, 2.4187128022, 2.414900355, 2.4111095369, 2.4071688205, 2.4032615721, 2.3993955627, 2.395164676, 2.3912272751, 2.3870118558, 2.3828617632, 2.3785987943, 2.374157533, 2.3698907271, 2.3654455617, 2.3612183332, 2.3566779345, 2.3521931991, 2.3478141725, 2.3433175609, 2.3388543949, 2.3342723697, 2.329891324, 2.3253212199, 2.3207966089, 2.3163165599, 2.3117237166, 2.307177335, 2.3028334454, 2.298258014, 2.2937453687, 2.2893637046, 2.2849271894, 2.2805777863, 2.2761499807, 2.2716621235, 2.2673755437, 2.2630329132, 2.2586768344, 2.2544075474, 2.2501640394, 2.2458122969, 2.2415976524, 2.2373722494, 2.233077094, 2.2289815024, 2.2248471081, 2.2206915766, 2.2165249735, 2.2125261799, 2.2084561139, 2.2044723034, 2.2005084753, 2.1966947839, 2.192776829, 2.1887145266, 2.1849389002, 2.1811656281, 2.1774653718, 2.1734953001, 2.1698662639, 2.1663491726, 2.1627112553, 2.1591341719, 2.1557029411, 2.152236864, 2.148811914, 2.1455652043, 2.1421615407, 2.138985686, 2.135702692, 2.1323651448, 2.1291328371, 2.126216419, 2.1231551096, 2.1201803982, 2.1171317324, 2.1141653359, 2.1113094017, 2.1085042506, 2.1056257412, 2.1027888209, 2.1000580639, 2.0972881615, 2.0947636962, 2.0919451639]


plt.plot(x, heap_sort_sgd_y, label="heap_sort_sgd", color="blue")
plt.plot(x, heap_sort_adam_y, label="heap_sort_adam", color="red")
plt.plot(x, insertion_sort_sgd_y, label="insertion_sort_sgd", color="blue")
plt.plot(x, insertion_sort_adam_y, label="insertion_sort_adam", color="red")
plt.plot(x, quick_sort_sgd_y, label="quick_sort_sgd", color="blue")
plt.plot(x, quick_sort_adam_y, label="quick_sort_adam", color="red")
plt.plot(x, top_down_merge_sort_sgd_y, label="top_down_merge_sort_sgd", color="blue")
plt.plot(x, top_down_merge_sort_adam_y, label="top_down_merge_sort_adam", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss (Cross-Entropy)")
plt.show()

