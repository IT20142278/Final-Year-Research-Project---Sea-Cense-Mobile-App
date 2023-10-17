import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:sea_cense/style.dart';
import 'package:sea_cense/viewmodels/sign_in_up_viewmodel.dart';
import 'package:sea_cense/views/home/home_view.dart';
import 'package:sea_cense/widgets/common_button_widget.dart';
import 'package:sea_cense/widgets/header_widget.dart';
import 'package:sea_cense/widgets/rounded_textbox_widget.dart';

class SignUpView extends StatelessWidget {
  const SignUpView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SizedBox(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        child: Consumer<SignInUpViewModel>(
          builder: (context, model, child) {
            return SingleChildScrollView(
              child: Column(
                children: [
                  const HeaderWidget(
                    text: "Create Account",
                    showBack: true,
                  ),
                  Container(
                    alignment: Alignment.centerLeft,
                    margin: const EdgeInsets.symmetric(horizontal: 50, vertical: 20),
                    child: const Text(
                      "Sign up",
                      style: topicStyle,
                    ),
                  ),
                  RoundedTextboxWidget(
                    controller: model.usernameController,
                    labelText: 'Username',
                  ),
                  RoundedTextboxWidget(
                    controller: model.passwordCntroller,
                    labelText: 'Password',
                    obscureText: true,
                  ),
                  RoundedTextboxWidget(
                    controller: model.confPasswordCntroller,
                    labelText: 'Confirm Password', obscureText: true,
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  CommonButtonWidget(
                    text: "Sign up",
                    size: MediaQuery.of(context).size.width / 1.3,
                    onTap: () {
                      model.signup(onSuccess: () {
                        Navigator.of(context).push(MaterialPageRoute(builder: (context) => const HomeView()));
                      });
                    },
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
