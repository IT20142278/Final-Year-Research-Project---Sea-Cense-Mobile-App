import 'package:flutter/material.dart';

class CustomDivider extends StatelessWidget {
  const CustomDivider({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white10,
      height: 2,
      width: double.infinity,
      margin: const EdgeInsets.symmetric(vertical: 10),
    );
  }
}
