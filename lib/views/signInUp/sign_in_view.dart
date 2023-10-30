import 'package:flutter/material.dart';
import 'package:sea_cense/style.dart';
import 'package:provider/provider.dart';
import 'package:sea_cense/viewmodels/sign_in_up_viewmodel.dart';
import 'package:sea_cense/views/home/home_view.dart';
import 'package:sea_cense/widgets/common_button_widget.dart';
import 'package:sea_cense/widgets/header_widget.dart';
import 'package:sea_cense/widgets/rounded_textbox_widget.dart';

class SignInView extends StatelessWidget {
  const SignInView({super.key});

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
                    text: "Welcome\nback",
                    showBack: true,
                  ),
                  Container(
                    alignment: Alignment.centerLeft,
                    margin: const EdgeInsets.symmetric(horizontal: 50, vertical: 20),
                    child: const Text(
                      "Sign in",
                      style: topicStyle,
                    ),
                  ),
                  RoundedTextboxWidget(
                    controller: model.usernameController,
                    labelText: 'Username',
                    keyboardType: TextInputType.emailAddress,
                  ),
                  RoundedTextboxWidget(
                    controller: model.passwordCntroller,
                    labelText: 'Password',
                    isPassword: true,
                  ),
                  const SizedBox(
                    height: 20,
                  ),
                  CommonButtonWidget(
                    text: "Sign in",
                    size: MediaQuery.of(context).size.width / 1.25,
                    onTap: () {
                      model.signin(onSuccess: () {
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
