import 'package:shared_preferences/shared_preferences.dart';

class SharedPreference {
  static setUser(String user) async {
    final SharedPreferences sharedPrefs = await SharedPreferences.getInstance();
    sharedPrefs.setString("user_info", user);
  }

  static Future<String?> getUser() async {
    final SharedPreferences sharedPrefs = await SharedPreferences.getInstance();
    return sharedPrefs.getString("user_info");
  }

  static clear() async {
    final SharedPreferences sharedPrefs = await SharedPreferences.getInstance();
    sharedPrefs.clear();
    sharedPrefs.reload();
  }
}
