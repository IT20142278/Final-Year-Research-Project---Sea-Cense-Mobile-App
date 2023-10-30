import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:sea_cense/viewmodels/sign_in_up_viewmodel.dart';
import 'package:sea_cense/views/signInUp/sign_in_view.dart';
import 'package:sea_cense/views/signInUp/sign_up_view.dart';
import 'package:sea_cense/widgets/common_button_widget.dart';
import 'package:sea_cense/widgets/header_widget.dart';

class OnboardingView extends StatelessWidget {
  const OnboardingView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SizedBox(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        child: Column(
          children: [
            const HeaderWidget(
              text: "Welcome to \nSea cense",
            ),
            SizedBox(
              height: MediaQuery.of(context).size.height / 5,
            ),
            CommonButtonWidget(
              text: "Sign in",
              size: MediaQuery.of(context).size.width / 1.3,
              onTap: () {
                Navigator.of(context).push(MaterialPageRoute(
                  builder: (context) => ChangeNotifierProvider(
                    create: (context) => SignInUpViewModel(),
                    child: const SignInView(),
                  ),
                ));
              },
            ),
            SizedBox(
              height: MediaQuery.of(context).size.height / 25,
            ),
            CommonButtonWidget(
              text: "Sign up",
              size: MediaQuery.of(context).size.width / 1.3,
              onTap: () {
                Navigator.of(context).push(MaterialPageRoute(
                  builder: (context) => ChangeNotifierProvider(
                    create: (context) => SignInUpViewModel(),
                    child: const SignUpView(),
                  ),
                ));
              },
            ),
          ],
        ),
      ),
    );
  }
}
