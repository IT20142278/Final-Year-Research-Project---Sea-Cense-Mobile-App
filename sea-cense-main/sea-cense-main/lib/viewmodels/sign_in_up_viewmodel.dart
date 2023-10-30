import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:sea_cense/models/user.dart';
import 'package:sea_cense/utils/navigation_service.dart';
import 'package:sea_cense/utils/storage.dart';
import 'package:sea_cense/utils/utils.dart';

class SignInUpViewModel extends ChangeNotifier {
  final GlobalKey<FormState> formKey = GlobalKey<FormState>();
  final TextEditingController usernameController = TextEditingController();
  final TextEditingController passwordCntroller = TextEditingController();
  final TextEditingController confPasswordCntroller = TextEditingController();

  List<User> userList = [];

  void signin({required VoidCallback onSuccess}) async {
    String? users = await SharedPreference.getUser();
    if (users != null) {
      userList = List<User>.from(json.decode(users).map((userJson) => User.fromJson(userJson)));
    }

    User user = User();
    user.username = usernameController.text;
    user.password = passwordCntroller.text;
    if (userList.isNotEmpty) {
      User? foundUser = userList.firstWhere(
        (value) => value.username!.toLowerCase() == user.username!.toLowerCase(),
        orElse: () => User(),
      );

      if (foundUser.username == "" || foundUser.username == null) {
        Utils.showSnackBar('User does not exist', NavigationService.navigatorKey.currentContext!);
      } else {
        if (foundUser.username!.toLowerCase() == user.username!.toLowerCase() &&
            foundUser.password == user.password) {
          onSuccess();
        } else {
          Utils.showSnackBar('Incorrect password', NavigationService.navigatorKey.currentContext!);
        }
      }
    } else {
      Utils.showSnackBar('User does not exist', NavigationService.navigatorKey.currentContext!);
    }
  }

  void signup({required VoidCallback onSuccess}) async {
    String? users = await SharedPreference.getUser();

    if (users == null || users == '') {
    } else {
      userList = List<User>.from(json.decode(users).map((userJson) => User.fromJson(userJson)));
    }

    if (usernameController.text.isEmpty ||
        passwordCntroller.text.isEmpty ||
        confPasswordCntroller.text.isEmpty) {
      Utils.showSnackBar('Fields cannot be empty', NavigationService.navigatorKey.currentContext!);
      return;
    }

    if (passwordCntroller.text == confPasswordCntroller.text) {
      User user = User();
      user.username = usernameController.text;
      user.password = passwordCntroller.text;

      if (userList.isNotEmpty) {
        User? foundUser = userList.firstWhere(
          (value) => value.username!.toLowerCase() == user.username!.toLowerCase(),
          orElse: () => User(),
        );

        if (foundUser.username == "" || foundUser.username == null) {
          userList.add(user);
          SharedPreference.setUser(jsonEncode(userList));
          onSuccess();
        } else {
          Utils.showSnackBar('User already exists', NavigationService.navigatorKey.currentContext!);
        }
      } else {
        userList.add(user);
        SharedPreference.setUser(jsonEncode(userList));
        onSuccess();
      }
    } else {
      Utils.showSnackBar('Passwords do not match', NavigationService.navigatorKey.currentContext!);
    }
  }

  disposeControllers() {
    usernameController.dispose();
    passwordCntroller.dispose();
    confPasswordCntroller.dispose();
    userList.clear();
  }
}
